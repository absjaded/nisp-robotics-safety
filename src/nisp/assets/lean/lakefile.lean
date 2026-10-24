import Lake
open Lake DSL

package nisp_axioms where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib NISPAxioms

