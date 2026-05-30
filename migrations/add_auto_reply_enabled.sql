-- Migration: Add auto_reply_enabled and auto_reply_prompt columns to channels table
-- Date: 2026-05-26
-- Description: Adds fields to enable/disable automatic replies via Instagram webhooks
--              and to customize the AI prompt for generating replies

ALTER TABLE channels 
ADD COLUMN IF NOT EXISTS auto_reply_enabled BOOLEAN DEFAULT FALSE;

ALTER TABLE channels 
ADD COLUMN IF NOT EXISTS auto_reply_prompt TEXT;

-- Update existing channels to have auto_reply disabled by default
UPDATE channels 
SET auto_reply_enabled = FALSE 
WHERE auto_reply_enabled IS NULL;
