# References

The ΔΔG training and evaluation data are based on FireProtDB:

- Musil, M., Borko, S., Planas-Iglesias, J., Lacko, D., Rosinska, M., Kabourek, P., Martins, L. O., Tataruch, M., Damborsky, J., Mazurenko, S., and Bednar, D. (2025). “FireProtDB 2.0: Large-scale manually curated database of protein stability data.” *Nucleic Acids Research*.

### Source for the experimental residue pKa values

The pKa pipeline is pretrained on experimentally measured residue pKas from PKAD:

- Pahari, S., Sun, L., and Alexov, E. (2019). “PKAD: a database of experimentally measured pKa values of ionizable groups in proteins.” *Database*, 2019, baz024. [doi:10.1093/database/baz024](https://doi.org/10.1093/database/baz024).

Residue numbering in PKAD follows each PDB entry's author numbering, so records are placed on their chain sequence using that entry's own mmCIF `pdbx_poly_seq_scheme` table, downloaded from the RCSB Protein Data Bank:

- Berman, H. M., Westbrook, J., Feng, Z., Gilliland, G., Bhat, T. N., Weissig, H., Shindyalov, I. N., and Bourne, P. E. (2000). “The Protein Data Bank.” *Nucleic Acids Research*, 28(1), 235–242. [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235).

### Source used in the whole-protein polymerase scan

The identification of the catalytic residues of the 3′→5′ proofreading exonuclease of human DNA polymerase δ, referred to in the limitations of [results/RESULTS_README.md](results/RESULTS_README.md), follows:

- Liu, D., Frederiksen, J. H., Liberti, S. E., Lützen, A., Keijzers, G., Pena-Diaz, J., and Rasmussen, L. J. (2017). “Human DNA polymerase delta double-mutant D316A;E318A interferes with DNA mismatch repair in vitro.” *Nucleic Acids Research*, 45(16), 9427–9440. [doi:10.1093/nar/gkx611](https://doi.org/10.1093/nar/gkx611).

### Source for the wild-type sequences

Wild-type sequences are taken from UniProt Swiss-Prot. The subset redistributed in
`data/Uniprot to sequence/` is derived from that release and is used under its licence, the
[Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/):

- The UniProt Consortium (2025). "UniProt: the Universal Protein Knowledgebase." *Nucleic Acids Research*. [https://www.uniprot.org](https://www.uniprot.org).

### Source for the pH-dependent term

The model obtains its pH dependence from the Wyman–Tanford linkage equation: it predicts a folded pKa for every residue, and the difference between the folded and unfolded protonation states gives the pH-dependent part of ΔΔG, as set out in [src/model/MODEL_README.md](src/model/MODEL_README.md#4-wymantanford-linkage-wymantanfordoutput). The idea of building the model around it came from the research in the source below, which uses the same linkage equation to compute pH-dependent stability from measured histidine pKas:

- Mosley, P. L., Daniels, K. G., and Oas, T. G. (2015). “Electrostatic Energetics of *Bacillus subtilis* Ribonuclease P Protein Determined by Nuclear Magnetic Resonance-Based Histidine pKa Measurements.” *Biochemistry*, 54(35), 5379–5388. [doi:10.1021/acs.biochem.5b00138](https://doi.org/10.1021/acs.biochem.5b00138).

### Sources for amino-acid pKa values

The model needs an unfolded-state pKa only for the side chains that actually titrate, and approximates it by the free amino-acid side-chain value. Those residues are marked `"titratable": true` in [amino_acid_info.json](<data/Amino acid information/amino_acid_info.json>) and carry a `pKa`.

Values for the ionisable side chains (C, D, E, H, K, R, and Y) were taken from:

- Vanderbilt University, [Amino Acids](https://www.vanderbilt.edu/AnS/Chemistry/Rizzo/stuff/AA/AminoAcids.html) (accessed 6 September 2026).
