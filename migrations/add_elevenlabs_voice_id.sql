-- Migration: Add elevenlabs_voice_id to channels table
-- Date: 2026-06-03
-- Description: Stores the ElevenLabs voice clone ID for each channel,
--              enabling TTS with the user's cloned voice in character videos.

ALTER TABLE channels
ADD COLUMN IF NOT EXISTS elevenlabs_voice_id VARCHAR(100);
