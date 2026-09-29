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
}

#[derive(Deserialize, Serialize, Clone)]
struct FunctionRange {
    image_start: u32,
    image_limit: u32,
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
    source_to_assembly: BTreeMap<u32, BTreeSet<u32>>,
    assembly_to_source: BTreeMap<u32, BTreeSet<u32>>,
    instructions: BTreeMap<u32, Instruction>,
    slots: BTreeMap<(u32, String), Slot>,
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
        let mut assembly_to_source: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut instructions = BTreeMap::new();
        for instruction in &program.instructions {
            let mut source_lines = BTreeSet::new();
            if let Some(slot) = slots.get(&(instruction.pc, instruction.slot.clone())) {
                for origin in &slot.origins {
                    for location in &origin.locations {
                        for frame in &location.frames {
                            if frame.path == "kernel.py" && frame.line_start > 0 {
                                // Only captured frames participate. Native annotation locations
                                // remain separate evidence in the instruction details.
                                for line in frame.line_start..=frame.line_end.max(frame.line_start)
                                {
                                    source_lines.insert(line);
                                }
                            }
                        }
                    }
                }
            }
            for line in &source_lines {
                source_to_assembly
                    .entry(*line)
                    .or_default()
                    .insert(instruction.line);
            }
            assembly_to_source.insert(instruction.line, source_lines);
            instructions.insert(instruction.line, instruction.clone());
        }
        Ok(Self {
            program,
            source_to_assembly,
            assembly_to_source,
            instructions,
            slots,
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
        let mapped: Vec<_> = self
            .assembly_to_source
            .iter()
            .filter(|(_, sources)| !sources.is_empty())
            .collect();
        let preferred_line = mapped.first().map(|(line, _)| **line).unwrap_or(1);
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
            "preferred_line": preferred_line,
            "pc_lines": pc_lines,
        })
        .to_string()
    }

    pub fn select_source(&self, line: u32) -> String {
        self.selection(
            self.source_to_assembly
                .get(&line)
                .cloned()
                .unwrap_or_default(),
        )
        .to_string()
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
        self.selection(lines).to_string()
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
    fn selection(&self, lines: BTreeSet<u32>) -> Value {
        let mut sources = BTreeSet::new();
        let details: Vec<_> = lines
            .iter()
            .filter_map(|line| {
                let instruction = self.instructions.get(line)?;
                if let Some(found) = self.assembly_to_source.get(line) {
                    sources.extend(found.iter().copied());
                }
                let slot = self.slots.get(&(instruction.pc, instruction.slot.clone()));
                Some(json!({"instruction":instruction,"source":slot}))
            })
            .collect();
        json!({"assembly_lines":lines,"source_lines":sources,"details":details})
    }
}
