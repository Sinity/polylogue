use super::*;
use std::io::{Read, Seek, SeekFrom, Write};

#[test]
fn missing_or_corrupt_empty_frames_never_install_blank_attachment_output() {
    for zip in [false, true] {
        for missing in [false, true] {
            // TestArchive imports actual provider bytes, then deletes the provider.
            let fixture = TestArchive::with_bytes(zip, b"");
            assert_eq!(fixture.descriptor()["state"], "available");
            let (extent, byte_root) = {
                let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(
                    &fixture.archive,
                ))
                .unwrap();
                let reader = pool.reader().unwrap();
                let blobs = reader.blob_reader().unwrap();
                let extent = blobs
                    .extent(&hashed_blobs::Hash(polylogue_model::BlobHash::of(b"").0))
                    .unwrap();
                (extent, blobs.root().to_path_buf())
            };
            assert!(extent.len > 0);
            let segment = byte_root.join(&extent.segment);
            let held = segment.with_extension("held-empty");
            let offset = extent.offset + extent.len - 1;
            let mut original = [0];
            if missing {
                std::fs::rename(&segment, &held).unwrap();
            } else {
                let mut file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&segment)
                    .unwrap();
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.read_exact(&mut original).unwrap();
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[original[0] ^ 1]).unwrap();
                file.sync_all().unwrap();
            }
            let directory = fixture._dir.path().join("output");
            std::fs::create_dir(&directory).unwrap();
            let existing = directory.join("existing.bin");
            let absent = directory.join("absent.bin");
            let sentinel = b"previous complete attachment output";
            std::fs::write(&existing, sentinel).unwrap();
            let results = [&existing, &absent].map(|destination| {
                cli(
                    &fixture.env(),
                    &[
                        "--json",
                        "attachment",
                        MESSAGE,
                        "0",
                        "--out",
                        destination.to_str().unwrap(),
                    ],
                )
            });
            // Restore the synthetic archive before assertions and daemon teardown.
            if missing {
                std::fs::rename(&held, &segment).unwrap();
            } else {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&segment)
                    .unwrap();
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&original).unwrap();
                file.sync_all().unwrap();
            }
            for result in results {
                assert!(!result.status.success(), "{result:?}");
            }
            assert_eq!(std::fs::read(&existing).unwrap(), sentinel);
            assert!(!absent.exists());
            assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
            // The same real CLI path can deliver the restored, verified body.
            let result = cli(
                &fixture.env(),
                &[
                    "--json",
                    "attachment",
                    MESSAGE,
                    "0",
                    "--out",
                    absent.to_str().unwrap(),
                ],
            );
            assert!(result.status.success(), "{result:?}");
            assert!(std::fs::read(&absent).unwrap().is_empty());
        }
    }
}
