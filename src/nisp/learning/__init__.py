# NISP learning package.
#
# Like src/nisp/interpret/, this package is INDEPENDENT from the runtime
# pipeline. Nothing in src/nisp/layers/ or src/nisp/core/ imports from here.
# These modules compute measurements over logged JSONL data — they never
# run inside the control loop.
