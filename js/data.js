// Reading the Stage A viewer export (exp1/pack_site.py). Everything is relative to the page: the build copies the
// fetched site/ directory to d/. The manifest describes planar typed arrays: for each field in layout order, round
// the byte offset up to the item size and read a C-order array (rows, columns), little endian. Files are gzip files
// (a file format, not HTTP Content-Encoding); a server that already decoded them is tolerated.
const ROOT = "d/";
const cache = new Map();
function once(key, fn) {
  if (!cache.has(key)) { const p = fn().catch(e => { cache.delete(key); throw e; }); cache.set(key, p); }
  return cache.get(key);
}
async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function gunzip(bytes) {
  if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;     // already decoded by the server
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export const manifest = () => once("manifest", async () => {
  const r = await fetch(ROOT + "manifest.json");
  if (!r.ok) throw new Error(`manifest: HTTP ${r.status}`);
  const m = await r.json();
  if (m.schema !== "stage-a-viewer-v1") throw new Error(`unexpected viewer data schema ${m.schema}`);
  if (m.byte_order !== "little") throw new Error("viewer data must be little endian");
  return m;
});

const TYPES = { "<u2": Uint16Array, "|u1": Uint8Array, "<f4": Float32Array, "<i2": Int16Array, "|i1": Int8Array };

// One asset by its manifest path: JSON, or {field: {data, columns}} typed planes, checked against the manifest.
export function asset(path) {
  return once("a:" + path, async () => {
    const m = await manifest(), spec = m.files[path];
    if (!spec) throw new Error(`${path}: not in the manifest`);
    const raw = await gunzip(await fetchBytes(`${ROOT}${m.data_directory}/${path}`));
    if (raw.length !== spec.decoded_bytes) throw new Error(`${path}: ${raw.length} bytes decoded, manifest says ${spec.decoded_bytes}`);
    if (spec.encoding === "json") return JSON.parse(new TextDecoder().decode(raw));
    if (spec.encoding !== "typed_arrays") throw new Error(`${path}: unknown encoding ${spec.encoding}`);
    const n = spec.shape[0], out = { rows: n }, buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    let off = 0;
    for (const f of m.layouts[spec.dtype]) {
      const T = TYPES[f.dtype];
      if (!T) throw new Error(`${path}: unsupported dtype ${f.dtype}`);
      off = Math.ceil(off / T.BYTES_PER_ELEMENT) * T.BYTES_PER_ELEMENT;
      const cols = f.columns || 1, count = n * cols;
      if (off + count * T.BYTES_PER_ELEMENT > buf.byteLength) throw new Error(`${path}: field ${f.name} runs past the end`);
      out[f.name] = { data: new T(buf, off, count), columns: cols, field: f };
      off += count * T.BYTES_PER_ELEMENT;
    }
    if (off !== buf.byteLength) throw new Error(`${path}: ${buf.byteLength - off} trailing bytes`);
    return out;
  });
}

export const json = url => once("j:" + url, async () => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
});

// Candidate pools: [[text, start, stop), ...] flattened in listed order. Returns a resolver index -> [text, position]
// and, per text, its first pool index and position range.
export function poolIndex(ranges) {
  const starts = [], byText = new Map();
  let off = 0;
  for (const [text, a, b] of ranges) { starts.push(off); byText.set(text, { offset: off, start: a, stop: b }); off += b - a; }
  const size = off;
  function locate(j) {
    if (!(j >= 0 && j < size)) return null;
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= j) lo = mid; else hi = mid - 1; }
    const [text, a] = ranges[lo];
    return [text, a + (j - starts[lo])];
  }
  return { locate, byText, size };
}
