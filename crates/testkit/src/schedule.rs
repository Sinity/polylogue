//! How files reach an archive: the order they are written in, each whole or
//! in appends cut at any byte, and between any two steps a new engine on the
//! same archive or the index deleted and built again.

use crate::{File, Home, delete_index, store_with};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// The most one job of the engine reads.
    pub chunk: u64,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Appends the file's bytes from where the last write ended up to `end`.
    Write { file: usize, end: usize },
    /// Runs the engine until nothing is due.
    Store,
    /// Drops the engine and starts another on the same archive.
    Restart,
    /// Drops the engine, deletes the index and stores again, building it
    /// from the stored bytes.
    Rebuild,
}

impl Schedule {
    /// Every file written whole, in order, then stored once.
    pub fn whole(files: &[File]) -> Self {
        let mut steps: Vec<Step> = files
            .iter()
            .enumerate()
            .map(|(file, f)| Step::Write {
                file,
                end: f.bytes.len(),
            })
            .collect();
        steps.push(Step::Store);
        Self {
            chunk: crate::WHOLE_FILES,
            steps,
        }
    }
}

impl Home {
    /// Writes `files` into the home and stores them as `schedule` says,
    /// ending with a store. Every file must be complete by the end. Returns
    /// the engine, for a caller that runs it again.
    pub fn deliver(
        &self,
        files: &[File],
        schedule: &Schedule,
    ) -> (polylogue_engine::Engine, polylogue_store::ArchiveLock) {
        let mut written = vec![0; files.len()];
        let mut engine = Some(self.engine(schedule.chunk));
        for step in &schedule.steps {
            match *step {
                Step::Write { file, end } => {
                    let f = &files[file];
                    self.append_bytes(&self.path_of(f), &f.bytes[written[file]..end]);
                    written[file] = end;
                }
                Step::Store => {
                    let (engine, _) = engine.as_mut().expect("an engine");
                    store_with(engine, &[]);
                }
                Step::Restart => {
                    drop(engine.take());
                    engine = Some(self.engine(schedule.chunk));
                }
                Step::Rebuild => {
                    drop(engine.take());
                    delete_index(&self.root);
                    let (mut e, lock) = self.engine(schedule.chunk);
                    store_with(&mut e, &[]);
                    engine = Some((e, lock));
                }
            }
        }
        for (file, f) in files.iter().enumerate() {
            assert_eq!(
                written[file],
                f.bytes.len(),
                "{} is incomplete",
                f.path.display()
            );
        }
        let (mut engine, lock) = engine.expect("an engine");
        store_with(&mut engine, &[]);
        (engine, lock)
    }

    /// Where `file` lies in this home.
    pub fn path_of(&self, file: &File) -> std::path::PathBuf {
        self.home.join(&file.path)
    }
}
