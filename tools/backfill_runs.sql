-- Backfill: link existing demo run_logs / call_decisions to a user-owned run.
--
-- Idempotent: only touches rows where run_ref IS NULL, and reuses an existing
-- run when one already exists for (user, run_id). Re-run safely after demos.
--
-- Target user: the OLDEST auth.users row by default (the single dev account in
-- this hackathon project). Change the SELECT below to target by email if needed.
--
-- Run with:
--   npx @insforge/cli db query "$(cat tools/backfill_runs.sql)" --unrestricted

DO $$
DECLARE
  uid UUID;
  r   RECORD;
  rid UUID;
BEGIN
  -- Pick the target user (oldest account). To target a specific person:
  --   SELECT id INTO uid FROM auth.users WHERE email = 'you@example.com';
  SELECT id INTO uid FROM auth.users ORDER BY created_at LIMIT 1;
  IF uid IS NULL THEN
    RAISE NOTICE 'No users found; nothing to backfill.';
    RETURN;
  END IF;

  -- Ensure the user has a profile (signups before the trigger existed lack one).
  INSERT INTO public.profiles (user_id, email, name)
  SELECT id, email, COALESCE(profile->>'name', email)
  FROM auth.users WHERE id = uid
  ON CONFLICT (user_id) DO NOTHING;

  -- One run per distinct unlinked run_id, then link its logs + decisions.
  FOR r IN
    SELECT run_id FROM public.run_logs       WHERE run_ref IS NULL
    UNION
    SELECT run_id FROM public.call_decisions WHERE run_ref IS NULL
  LOOP
    SELECT id INTO rid FROM public.runs
      WHERE user_id = uid AND name = r.run_id LIMIT 1;

    IF rid IS NULL THEN
      INSERT INTO public.runs (user_id, name, status)
      VALUES (uid, r.run_id, 'finished')
      RETURNING id INTO rid;
    END IF;

    UPDATE public.run_logs       SET run_ref = rid WHERE run_id = r.run_id AND run_ref IS NULL;
    UPDATE public.call_decisions SET run_ref = rid WHERE run_id = r.run_id AND run_ref IS NULL;
  END LOOP;

  RAISE NOTICE 'Backfill complete for user %', uid;
END $$;
