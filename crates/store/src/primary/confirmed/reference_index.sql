CREATE INDEX IF NOT EXISTS confirmed_event_reference ON confirmed_event(reference,kind,state,committed,source_order,event);
