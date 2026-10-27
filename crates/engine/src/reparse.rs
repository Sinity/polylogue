//! Re-read stored bytes into scratch, then replace a chain in one transaction.
use crate::*;

impl Engine {
    pub(crate) fn replace_reading(
        &mut self,
        mut plan: Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let Change::Reparse { chain } = plan.change else {
            return Err(std::io::Error::other("a replacement needs a stored chain").into());
        };
        if !self.load_stored(&mut plan)? {
            return Ok(());
        }
        self.replacement = Some(polylogue_store::writer::ChainReading::new()?);
        let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<(), EngineError> {
                if let Some(polylogue_sources::Parse::Export(mut parser)) =
                    (plan.origin.open)(&plan.path)
                {
                    self.execute_export(&mut plan, &mut *parser, pass)?;
                } else {
                    while !plan.finished() {
                        let job = plan.next_job();
                        pass.bytes_read += job.end - job.start;
                        let (batch, done) = pool::run(job, &self.archive, &self.blobs, false);
                        if let Some(error) = &done.error {
                            return Err(std::io::Error::other(error.clone()).into());
                        }
                        let Some(mut batch) = batch else {
                            return Err(
                                std::io::Error::other("a replacement returned no rows").into()
                            );
                        };
                        for step in &mut batch.batch.parse {
                            if step.parsed_upto == step.start && !step.at_end {
                                continue;
                            }
                            self.replacement
                                .as_mut()
                                .expect("reading started")
                                .push(step)?;
                        }
                        batch.batch.parse.clear();
                        if !batch.batch.blobs.is_empty() {
                            for result in self
                                .writer
                                .sender()?
                                .store_group(vec![batch], Arc::clone(&self.blobs))?
                            {
                                result?;
                            }
                        }
                        plan.apply(done, true);
                    }
                }
                Ok(())
            },
        ));
        let reading = self.replacement.take().expect("reading started");
        let parsed = parsed.unwrap_or_else(|_| {
            Err(std::io::Error::other("the replacement parser panicked").into())
        });
        if let Err(error) = parsed {
            plan.failed = true;
            self.finish_plan(plan);
            return Err(error);
        }
        let written =
            self.writer
                .sender()?
                .replace_chain(plan.input.clone(), chain, plan.size, reading);
        match written {
            Ok(written) => {
                pass.messages_written += written;
                self.finish_plan(plan);
                Ok(())
            }
            Err(error) => {
                plan.failed = true;
                self.finish_plan(plan);
                Err(error.into())
            }
        }
    }
}
