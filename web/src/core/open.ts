/**
 * One entry point for both file kinds: looks at the container header and hands the file to the dump or the
 * recording reader. The viewer and the player used to be separate programs; here a single call decides.
 */

import { Container, ContainerError } from './container.ts';
import { parseDemo, type TraceDemo } from './demo.ts';
import { parseDump, type TraceDump } from './trace.ts';

/** A parsed trace file of either kind; `.kind` tells which. */
export type Trace = TraceDump | TraceDemo;

/**
 * Parses a trace file of either kind.
 * @param buffer the whole file
 * @param name file name for messages
 */
export function openTrace(buffer: ArrayBuffer, name: string): Trace {
  const container = new Container(buffer, name);
  const kind = container.header.kind ?? 'dump';
  if (kind === 'dump') return parseDump(container);
  if (kind === 'demo') return parseDemo(container);
  throw new ContainerError(`${name}: unknown recording kind ${JSON.stringify(kind)} (expected "dump" or "demo")`);
}
