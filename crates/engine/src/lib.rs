//! Browser-side index over tpuasm's explicit (program, PC, slot) identities.
//! Instruction coordinates come from tpuasm's parser, never from guessed text.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen::prelude::*;

#[derive(Deserialize, Serialize, Clone)]
struct Instruction {
    pc: u32,
    slot: String,
    line: u32,
    mnemonic: String,
}

#[derive(Deserialize)]
struct Bundle {
    pc: u32,
    line: u32,
}

#[derive(Deserialize, Serialize, Clone)]
struct Frame {
    path: String,
    line_start: u32,
    line_end: u32,
    col_start: u32,
    col_end: u32,
    #[serde(default)]
    function_name: String,
}

#[derive(Deserialize, Serialize, Clone)]
struct Location {
    frames: Vec<Frame>,
    primitive: String,
    scope_stack: Vec<String>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Origin {
    hlo_name: String,
    hlo_module_name: String,
    hlo_module_id: String,
    llo_ordinal: u32,
    locations: Vec<Location>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Slot {
    image_pc: u32,
    slot: String,
    compiler_annotation: String,
    annotation_locations: Vec<Frame>,
    origins: Vec<Origin>,
    function_symbols: Vec<u32>,
    source_frames: Vec<Frame>,
    source_kind: String,
}

/// One captured call chain, innermost frame first, as tpuasm's listing prints
/// `inner <- caller`. Every scope/ordinal recorded for the same chain is merged.
#[derive(Serialize, Clone)]
struct Chain {
    frames: Vec<Frame>,
    labels: Vec<Label>,
}

/// Scopes that share exactly the same LLO ordinals, e.g. `ne, lt, and; LLO 116`.
#[derive(Serialize, Clone)]
struct Label {
    scopes: Vec<String>,
    ordinals: Vec<u32>,
}

/// Display form of one slot's evidence. `compiler` holds native `loc(...)`
/// positions absent from every chain; `notes` is the remaining native text.
#[derive(Serialize, Clone, Default)]
struct SourceView {
    chains: Vec<Chain>,
    compiler: Vec<Frame>,
    notes: Vec<String>,
}

type Position = (String, u32, u32, u32, u32);

fn position(frame: &Frame) -> Position {
    (
        frame.path.clone(),
        frame.line_start,
        frame.line_end,
        frame.col_start,
        frame.col_end,
    )
}

/// A chain being collected: frame positions, displayed frames, and ordinals per scope.
type PendingChain = (Vec<Position>, Vec<Frame>, Vec<(String, Vec<u32>)>);

fn source_view(slot: &Slot) -> SourceView {
    let mut chains: Vec<PendingChain> = Vec::new();
    for origin in &slot.origins {
        for location in &origin.locations {
            let frames: Vec<Frame> = location.frames.iter().rev().cloned().collect();
            let key: Vec<Position> = frames.iter().map(position).collect();
            let index = match chains.iter().position(|(existing, _, _)| *existing == key) {
                Some(index) => index,
                None => {
                    chains.push((key, frames.clone(), Vec::new()));
                    chains.len() - 1
                }
            };
            let (_, shown, scopes) = &mut chains[index];
            for (shown, frame) in shown.iter_mut().zip(&frames) {
                if shown.function_name.is_empty() && !frame.function_name.is_empty() {
                    *shown = frame.clone();
                }
            }
            let scope = location
                .scope_stack
                .iter()
                .chain([&location.primitive])
                .filter(|part| !part.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("/");
            let ordinals = match scopes.iter().position(|(name, _)| *name == scope) {
                Some(index) => &mut scopes[index].1,
                None => {
                    scopes.push((scope, Vec::new()));
                    &mut scopes.last_mut().unwrap().1
                }
            };
            if !ordinals.contains(&origin.llo_ordinal) {
                ordinals.push(origin.llo_ordinal);
            }
        }
    }
    let displayed: BTreeSet<Position> = chains.iter().flat_map(|(key, _, _)| key.clone()).collect();
    let mut compiler: Vec<Frame> = Vec::new();
    for frame in &slot.annotation_locations {
        if !displayed.contains(&position(frame))
            && !compiler
                .iter()
                .any(|shown| position(shown) == position(frame))
        {
            compiler.push(frame.clone());
        }
    }
    let chains = chains
        .into_iter()
        .map(|(_, frames, scopes)| {
            let mut labels: Vec<Label> = Vec::new();
            for (scope, ordinals) in scopes {
                match labels.iter_mut().find(|label| label.ordinals == ordinals) {
                    Some(label) => label.scopes.push(scope),
                    None => labels.push(Label {
                        scopes: vec![scope],
                        ordinals,
                    }),
                }
            }
            for label in &mut labels {
                label.scopes.retain(|scope| !scope.is_empty());
            }
            Chain { frames, labels }
        })
        .collect();
    SourceView {
        chains,
        compiler,
        notes: annotation_notes(&slot.compiler_annotation),
    }
}

/// Native annotations join entries with " :: ". Drop `loc(...)` entries, which
/// tpuasm already parsed into `annotation_locations`, and keep all other text.
fn annotation_notes(text: &str) -> Vec<String> {
    let mut notes: Vec<String> = Vec::new();
    for part in text.split(" :: ") {
        let mut rest = part;
        let mut kept = String::new();
        while let Some(start) = rest.find("loc(\"") {
            kept.push_str(&rest[..start]);
            match location_length(&rest[start..]) {
                Some(length) => rest = &rest[start + length..],
                None => {
                    kept.push_str(&rest[start..start + 5]);
                    rest = &rest[start + 5..];
                }
            }
        }
        kept.push_str(rest);
        let kept = kept.trim();
        if !kept.is_empty() && !notes.iter().any(|note| note == kept) {
            notes.push(kept.to_string());
        }
    }
    notes
}

/// Length of `loc("path":L:C)` or `loc("path":L:C to [L]:C)` at the start of `text`.
fn location_length(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = 5;
    while index < bytes.len() && bytes[index] != b'"' {
        index += if bytes[index] == b'\\' { 2 } else { 1 };
    }
    index += 1;
    let digits = |index: &mut usize| {
        let start = *index;
        while *index < bytes.len() && bytes[*index].is_ascii_digit() {
            *index += 1;
        }
        *index - start
    };
    for _ in 0..2 {
        if bytes.get(index) != Some(&b':') {
            return None;
        }
        index += 1;
        if digits(&mut index) == 0 {
            return None;
        }
    }
    if text[index..].starts_with(" to ") {
        index += 4;
        digits(&mut index);
        if bytes.get(index) != Some(&b':') {
            return None;
        }
        index += 1;
        if digits(&mut index) == 0 {
            return None;
        }
    }
    (bytes.get(index) == Some(&b')')).then_some(index + 1)
}

/// Source lines an instruction maps to. `primary` is the innermost kernel.py
/// frame of each chain plus native locations; `callers` are the other frames.
#[derive(Default)]
struct Lines {
    primary: BTreeSet<u32>,
    callers: BTreeSet<u32>,
    ranges: Vec<Frame>,
}

fn frame_lines(frame: &Frame) -> impl Iterator<Item = u32> {
    frame.line_start..=frame.line_end.max(frame.line_start)
}

fn kernel_frame(frame: &Frame) -> bool {
    frame.path == "kernel.py" && frame.line_start > 0
}

fn view_lines(view: &SourceView) -> Lines {
    let mut lines = Lines::default();
    let mut callers = BTreeSet::new();
    for chain in &view.chains {
        let mut kernel = chain.frames.iter().filter(|frame| kernel_frame(frame));
        if let Some(frame) = kernel.next() {
            lines.primary.extend(frame_lines(frame));
            lines.ranges.push(frame.clone());
        }
        for frame in kernel {
            callers.extend(frame_lines(frame));
        }
    }
    for frame in view.compiler.iter().filter(|frame| kernel_frame(frame)) {
        lines.primary.extend(frame_lines(frame));
        lines.ranges.push(frame.clone());
    }
    lines.callers = callers.difference(&lines.primary).copied().collect();
    lines
}

#[derive(Deserialize, Serialize, Clone)]
struct FunctionRange {
    image_start: u32,
    image_limit: u32,
}

#[derive(Serialize)]
struct LineRange {
    start: u32,
    end: u32,
}

#[derive(Deserialize, Serialize, Clone)]
struct Function {
    symbol_id: u32,
    display_name: String,
    ranges: Vec<FunctionRange>,
}

#[derive(Deserialize)]
struct SourceMap {
    bundle_count: u32,
    status: String,
    slots: Vec<Slot>,
    functions: Vec<Function>,
    diagnostics: Vec<String>,
}

#[derive(Deserialize)]
struct Program {
    id: String,
    assembly: String,
    instructions: Vec<Instruction>,
    bundles: Vec<Bundle>,
    source_map: SourceMap,
}

#[wasm_bindgen]
pub struct SourceIndex {
    program: Program,
    /// Source line -> assembly lines whose innermost frame is that line.
    source_to_assembly: BTreeMap<u32, BTreeSet<u32>>,
    /// Source line -> assembly lines that reach it only as a caller frame.
    caller_to_assembly: BTreeMap<u32, BTreeSet<u32>>,
    assembly_lines: BTreeMap<u32, Lines>,
    instructions: BTreeMap<u32, Instruction>,
    slots: BTreeMap<(u32, String), Slot>,
    views: BTreeMap<u32, SourceView>,
}

#[wasm_bindgen]
impl SourceIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(input: &str) -> Result<SourceIndex, JsError> {
        let program: Program = serde_json::from_str(input)?;
        let slots: BTreeMap<_, _> = program
            .source_map
            .slots
            .iter()
            .cloned()
            .map(|slot| ((slot.image_pc, slot.slot.clone()), slot))
            .collect();
        let mut source_to_assembly: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut caller_to_assembly: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut assembly_lines = BTreeMap::new();
        let mut instructions = BTreeMap::new();
        let mut views = BTreeMap::new();
        for instruction in &program.instructions {
            // Both levels come from tpuasm's recorded frames and native
            // locations; a caller frame (e.g. a pl.loop decorator) is context,
            // not the operation that produced this instruction.
            let view = slots
                .get(&(instruction.pc, instruction.slot.clone()))
                .map(source_view)
                .unwrap_or_default();
            let lines = view_lines(&view);
            for line in &lines.primary {
                source_to_assembly
                    .entry(*line)
                    .or_default()
                    .insert(instruction.line);
            }
            for line in &lines.callers {
                caller_to_assembly
                    .entry(*line)
                    .or_default()
                    .insert(instruction.line);
            }
            assembly_lines.insert(instruction.line, lines);
            instructions.insert(instruction.line, instruction.clone());
            views.insert(instruction.line, view);
        }
        Ok(Self {
            program,
            source_to_assembly,
            caller_to_assembly,
            assembly_lines,
            instructions,
            slots,
            views,
        })
    }

    pub fn summary(&self) -> String {
        let mappings: Vec<_> = self
            .source_to_assembly
            .iter()
            .map(|(line, assembly)| json!({"line": line, "assembly_lines": assembly}))
            .collect();
        let functions: Vec<_> = self
            .program
            .source_map
            .functions
            .iter()
            .map(|function| {
                let lines: Vec<_> = self
                    .program
                    .instructions
                    .iter()
                    .filter(|instruction| {
                        function.ranges.iter().any(|range| {
                            instruction.pc >= range.image_start
                                && instruction.pc < range.image_limit
                        })
                    })
                    .map(|instruction| instruction.line)
                    .collect();
                json!({"id": function.symbol_id, "name": function.display_name, "lines": lines})
            })
            .collect();
        let mut function_spans: Vec<(u32, u32)> = self
            .program
            .source_map
            .functions
            .iter()
            .flat_map(|function| &function.ranges)
            .map(|range| (range.image_start, range.image_limit))
            .collect();
        function_spans.sort_unstable();
        let mut merged_spans: Vec<(u32, u32)> = Vec::new();
        for (start, limit) in function_spans {
            if let Some((_, previous_limit)) = merged_spans.last_mut()
                && start <= *previous_limit
            {
                *previous_limit = (*previous_limit).max(limit);
            } else {
                merged_spans.push((start, limit));
            }
        }
        let bundle_lines: BTreeMap<_, _> = self
            .program
            .bundles
            .iter()
            .map(|bundle| (bundle.pc, bundle.line))
            .collect();
        let assembly_line_count = self.program.assembly.lines().count() as u32;
        let kernel_ranges: Vec<_> = merged_spans
            .iter()
            .filter_map(|(start, limit)| {
                let start_line = *bundle_lines.get(start)?;
                let end_line = bundle_lines
                    .get(limit)
                    .map(|line| line.saturating_sub(1))
                    .unwrap_or(assembly_line_count);
                Some(LineRange {
                    start: start_line,
                    end: end_line,
                })
            })
            .collect();
        let mapped: Vec<_> = self
            .assembly_lines
            .iter()
            .filter(|(_, lines)| !lines.primary.is_empty())
            .collect();
        // Inline hint per instruction: its primary source lines and the first
        // recorded primitive. Native-location-only instructions have no primitive.
        let hints: Vec<_> = mapped
            .iter()
            .map(|(line, lines)| {
                let view = &self.views[*line];
                let primitive = view
                    .chains
                    .iter()
                    .flat_map(|chain| &chain.labels)
                    .flat_map(|label| &label.scopes)
                    .map(|scope| scope.rsplit('/').next().unwrap_or(scope))
                    .next();
                json!({"line": line, "source_lines": lines.primary, "primitive": primitive, "captured": !view.chains.is_empty()})
            })
            .collect();
        let preferred_line = mapped
            .first()
            .map(|(line, _)| **line)
            .or_else(|| kernel_ranges.first().map(|range| range.start))
            .unwrap_or(1);
        let pc_lines: Vec<_> = self
            .program
            .bundles
            .iter()
            .map(|bundle| json!({"line":bundle.line,"pc":bundle.pc}))
            .collect();
        json!({
            "program_id": self.program.id,
            "bundle_count": self.program.source_map.bundle_count,
            "instruction_count": self.instructions.len(),
            "mapped_count": mapped.len(),
            "status": self.program.source_map.status,
            "diagnostics": self.program.source_map.diagnostics,
            "mappings": mappings,
            "functions": functions,
            "kernel_ranges": kernel_ranges,
            "hints": hints,
            "preferred_line": preferred_line,
            "pc_lines": pc_lines,
        })
        .to_string()
    }

    pub fn select_source(&self, line: u32) -> String {
        let lines = self
            .source_to_assembly
            .get(&line)
            .cloned()
            .unwrap_or_default();
        let callers = self
            .caller_to_assembly
            .get(&line)
            .map(|found| found.difference(&lines).copied().collect())
            .unwrap_or_default();
        self.selection(lines, callers).to_string()
    }

    pub fn select_assembly(&self, line: u32) -> String {
        let lines = if self.instructions.contains_key(&line) {
            BTreeSet::from([line])
        } else if let Some(bundle) = self
            .program
            .bundles
            .iter()
            .find(|bundle| bundle.line == line)
        {
            self.instructions
                .values()
                .filter(|instruction| instruction.pc == bundle.pc)
                .map(|instruction| instruction.line)
                .collect()
        } else {
            BTreeSet::new()
        };
        self.selection(lines, BTreeSet::new()).to_string()
    }

    pub fn search(&self, query: &str) -> String {
        let query = query.trim().to_lowercase();
        let lines: Vec<u32> = if query.is_empty() {
            Vec::new()
        } else {
            self.program
                .assembly
                .lines()
                .enumerate()
                .filter(|(_, text)| text.to_lowercase().contains(&query))
                .map(|(index, _)| index as u32 + 1)
                .collect()
        };
        json!(lines).to_string()
    }
}

impl SourceIndex {
    /// `lines` are selected instructions; `callers` are instructions reached
    /// only through a caller frame of the selected source line.
    fn selection(&self, lines: BTreeSet<u32>, callers: BTreeSet<u32>) -> Value {
        let mut sources = BTreeSet::new();
        let mut caller_sources = BTreeSet::new();
        let mut ranges: Vec<&Frame> = Vec::new();
        let details: Vec<_> = lines
            .iter()
            .filter_map(|line| {
                let instruction = self.instructions.get(line)?;
                if let Some(found) = self.assembly_lines.get(line) {
                    sources.extend(found.primary.iter().copied());
                    caller_sources.extend(found.callers.iter().copied());
                    for frame in &found.ranges {
                        if !ranges
                            .iter()
                            .any(|shown| position(shown) == position(frame))
                        {
                            ranges.push(frame);
                        }
                    }
                }
                let slot = self.slots.get(&(instruction.pc, instruction.slot.clone()));
                Some(json!({"instruction":instruction,"source":slot,"view":self.views.get(line)}))
            })
            .collect();
        let caller_sources: BTreeSet<u32> = caller_sources.difference(&sources).copied().collect();
        json!({
            "assembly_lines": lines,
            "caller_assembly_lines": callers,
            "source_lines": sources,
            "caller_source_lines": caller_sources,
            "source_ranges": ranges,
            "details": details,
        })
    }
}
