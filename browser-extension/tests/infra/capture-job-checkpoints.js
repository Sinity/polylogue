import { createHash } from "node:crypto";

/** Receiver checkpoint state shared by the background RPC fixtures. */
export function checkpointReceiverState() {
  const jobs = new Map();
  let generation = 0;
  return {
    clear() { generation += 1; jobs.clear(); },
    job(value) {
      if (!value?.job_id) return value;
      const previous = jobs.get(value.job_id);
      const current = { ...previous, ...value };
      if (previous?.checkpoint_digest) {
        current.checkpoint = previous.checkpoint;
        current.checkpoint_sequence = previous.checkpoint_sequence;
        current.checkpoint_digest = previous.checkpoint_digest;
      } else if (value.checkpoint) {
        current.checkpoint_sequence = value.checkpoint.sequence;
        current.checkpoint_digest = value.checkpoint.digest;
      }
      jobs.set(value.job_id, current);
      return current;
    },
    async checkpoint(value, options) {
      const capturedGeneration = generation;
      const descriptor = JSON.parse(options.headers["X-Polylogue-Checkpoint"]);
      const bytes = await options.body.text();
      if (generation !== capturedGeneration) throw new Error("fixture_checkpoint_receiver_replaced");
      const digest = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
      if (descriptor.digest !== digest) throw new Error("fixture_checkpoint_digest_mismatch");
      const current = { ...jobs.get(value.job_id), ...value,
        checkpoint_sequence: descriptor.sequence, checkpoint_digest: digest,
        checkpoint: { sequence: descriptor.sequence, digest, artifact_ref: digest, size_bytes: globalThis.Buffer.byteLength(bytes) } };
      jobs.set(value.job_id, current);
      return { job: current, receipt: { checkpoint_sequence: descriptor.sequence, checkpoint_digest: digest }, bytes };
    },
  };
}
