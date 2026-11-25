use super::*;

fn index() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn
}

#[test]
fn pieces_lie_one_after_another_in_groups_and_a_long_piece_is_alone() {
    let mut packer = Packer::default();
    let short: Vec<Vec<u8>> = (0..40)
        .map(|n| format!("{{\"synthetic\":{n},\"walrus\":\"counted the tide\"}}").into_bytes())
        .collect();
    let long = vec![b'x'; GROUP_BYTES + 1];
    let mut placed = Vec::new();
    for (n, piece) in short.iter().enumerate() {
        placed.push(packer.add(piece).unwrap());
        if n == 19 {
            placed.push(packer.add(&long).unwrap());
        }
    }
    let empty = packer.add(b"").unwrap();
    let groups = packer.finish().unwrap();
    // Twenty pieces, then the long one alone, then the other twenty and the
    // empty one.
    assert_eq!(groups.0.len(), 3);
    assert_eq!(placed[0].group, 0);
    assert_eq!(placed[19].group, 0);
    assert_eq!((placed[20].group, placed[20].start), (1, 0));
    assert_eq!((placed[21].group, placed[21].start), (2, 0));
    assert_eq!(placed[1].start, placed[0].length);
    assert_eq!(empty.group, 2);
    let mut pieces: Vec<&[u8]> = short.iter().map(Vec::as_slice).collect();
    pieces.insert(20, &long);
    for (piece, placed) in pieces.iter().zip(&placed) {
        assert_eq!(groups.read(*placed).unwrap(), *piece);
    }
    assert_eq!(groups.read(empty).unwrap(), b"");
}

#[test]
fn a_group_alone_holding_an_empty_piece_is_made() {
    let mut packer = Packer::default();
    let placed = packer.add(b"").unwrap();
    let groups = packer.finish().unwrap();
    assert_eq!(groups.0.len(), 1);
    assert_eq!(groups.read(placed).unwrap(), b"");
}

#[test]
fn parts_read_back_as_written_and_damaged_parts_refuse() {
    let parts: [&[u8]; 4] = [b"{\"head\":1}", b"", &[b'a'; 300], b"[]"];
    let joined = join_parts(&parts);
    let ranges = super::parts(&joined).unwrap();
    assert_eq!(ranges.len(), 4);
    for (part, range) in parts.iter().zip(ranges) {
        assert_eq!(&joined[range], *part);
    }
    assert!(super::parts(&joined[..joined.len() - 1]).is_err());
    let mut longer = joined.clone();
    longer.push(0);
    assert!(super::parts(&longer).is_err());
    assert!(super::parts(&[]).is_err());
}

#[test]
fn a_stored_group_is_one_row_per_body_and_leaves_once_nothing_names_it() {
    let conn = index();
    let mut packer = Packer::default();
    let placed = packer.add(b"{\"fields\":true}").unwrap();
    let groups = packer.finish().unwrap();
    let first = Stored::new(&groups).place(&conn, placed).unwrap();
    let again = Stored::new(&groups).place(&conn, placed).unwrap();
    assert_eq!(first, again);
    let count = |conn: &Connection| -> i64 {
        conn.query_row("SELECT count(*) FROM content_group", [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(count(&conn), 1);
    let mut reader = Reader::default();
    assert_eq!(reader.bytes(&conn, first).unwrap(), b"{\"fields\":true}");
    // A text names its group: it stays.
    conn.execute(
        "INSERT INTO text(hash,normalized_hash,group_id,start,length) VALUES (x'00',x'00',?1,0,1)",
        [first.group],
    )
    .unwrap();
    release(&conn, first.group).unwrap();
    assert_eq!(count(&conn), 1);
    conn.execute("DELETE FROM text", []).unwrap();
    release(&conn, first.group).unwrap();
    assert_eq!(count(&conn), 0);
    // The key of a removed group is never given to another.
    let mut packer = Packer::default();
    let placed = packer.add(b"{\"other\":true}").unwrap();
    let groups = packer.finish().unwrap();
    let second = Stored::new(&groups).place(&conn, placed).unwrap();
    assert!(second.group > first.group);
}
