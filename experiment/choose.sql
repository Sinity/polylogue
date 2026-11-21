INSERT INTO selected(session,id,observation)
VALUES(?1,?2,(
 SELECT o.pk FROM version v JOIN observation o ON o.version=v.pk JOIN file f ON f.id=o.file
 WHERE v.session=?1 AND v.id=?2
 ORDER BY o.priority DESC,f.name ASC,v.hash ASC LIMIT 1))
ON CONFLICT(session,id) DO UPDATE SET observation=excluded.observation
WHERE selected.observation IS NOT excluded.observation;
