export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string') {
    throw new TypeError(`response field ${key} must be a string`);
  }
  return value;
}

export function nullableString(
  record: Record<string, unknown>,
  key: string,
): string | null {
  const value = record[key];
  if (value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw new TypeError(`response field ${key} must be a string or null`);
  }
  return value;
}

export function requiredNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`response field ${key} must be a finite number`);
  }
  return value;
}

export function nullableNumber(
  record: Record<string, unknown>,
  key: string,
): number | null {
  const value = record[key];
  if (value === null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`response field ${key} must be a finite number or null`);
  }
  return value;
}

export function requiredBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (typeof value !== 'boolean') {
    throw new TypeError(`response field ${key} must be a boolean`);
  }
  return value;
}

export function optionalBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (value === undefined || value === null) {
    return false;
  }
  return requiredBoolean(record, key);
}
