-- A physical member interpretation belongs to this exact original container.
-- The member observation owns the relationship in its publication unit.
CREATE TABLE observation_container (
    observation_id TEXT PRIMARY KEY REFERENCES observation(id),
    container_input TEXT NOT NULL REFERENCES input(id),
    member_index INTEGER NOT NULL CHECK(member_index>=0)
) STRICT;
CREATE INDEX observation_container_input ON observation_container(container_input);
