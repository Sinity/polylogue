CREATE INDEX IF NOT EXISTS message_model ON message(model) WHERE model IS NOT NULL;
CREATE INDEX IF NOT EXISTS message_at ON message(at, id);
CREATE INDEX IF NOT EXISTS message_session_at ON message(session_pk, at, id);
CREATE INDEX IF NOT EXISTS message_model_at ON message(model, at, id);
CREATE INDEX IF NOT EXISTS message_session_model_at ON message(session_pk, model, at, id);
CREATE INDEX IF NOT EXISTS event_at ON event(at, id);
