from pathlib import Path
from zipfile import ZipFile
import csv, hashlib, itertools, json, math, shlex, statistics

ROOT=Path(__file__).resolve().parent
AA3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
CODONS=dict(zip((''.join(c) for c in itertools.product('TCAG',repeat=3)),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
def sequence(p):return ''.join(x.strip() for x in p.read_text().splitlines() if not x.startswith('>')).upper()
def atoms(text):
    headers=[]; rows=[]; capture=False
    for line in text.splitlines():
        if line.startswith('_atom_site.'):
            headers.append(line.split()[0]);capture=True
        elif capture and line.startswith(('ATOM ','HETATM ')):
            values=shlex.split(line); assert len(values)==len(headers)
            rows.append(dict(zip(headers,values)))
        elif capture and rows: break
    return rows
def main():
    checked=0
    for line in (ROOT/'checksums.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1);p=(ROOT/relative).resolve()
        assert p.is_relative_to(ROOT),relative
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,relative
        checked+=1
    with (ROOT/'audit/structure_models.tsv').open() as f:
        expected_models={(r['protein_id'],int(r['model_index'])):r for r in csv.DictReader(f,delimiter='\t')}
    model_count=0
    for var in ['wt','alt']:
        pep=sequence(ROOT/f'canonical_inputs/FLM_{var}_pep.fa')
        cds=sequence(ROOT/f'canonical_inputs/FLM_{var}_CDS.fa')
        translation=''.join(CODONS.get(cds[i:i+3],'X') for i in range(0,len(cds)-2,3)).split('*')[0]
        assert translation==pep,(var,'CDS to peptide')
        parent=ROOT.parent/'input'/f'FLM_{var}_CDS.fa'
        if parent.exists():assert sequence(parent)==cds,(var,'parent repository input differs')
        with ZipFile(ROOT/f'raw_af3/fold_flm_{var}.zip') as z:
            request=json.loads(z.read(f'fold_flm_{var}_job_request.json'))[0]
            assert request['sequences'][0]['proteinChain']['sequence']==pep
            for i in range(5):
                native=atoms(z.read(f'fold_flm_{var}_model_{i}.cif').decode())
                pdb=[l for l in (ROOT/f'converted_pdb/fold_flm_{var}_model_{i}.pdb').read_text().splitlines() if l.startswith('ATOM  ')]
                assert len(native)==len(pdb)
                for r,p in zip(native,pdb):
                    assert r['_atom_site.label_atom_id']==p[12:16].strip()
                    assert r['_atom_site.label_comp_id']==p[17:20].strip()
                    assert int(r['_atom_site.label_seq_id'])==int(p[22:26])
                    assert float(r['_atom_site.B_iso_or_equiv'])==float(p[60:66])
                    for axis,start,end in [('x',30,38),('y',38,46),('z',46,54)]:assert float(r[f'_atom_site.Cartn_{axis}'])==float(p[start:end])
                ca=[p for p in pdb if p[12:16].strip()=='CA']
                assert ''.join(AA3[p[17:20].strip()] for p in ca)==pep
                record=expected_models[(f'FLM_{var}',i)]
                atom_mean=statistics.mean(float(p[60:66]) for p in pdb)
                assert math.isclose(atom_mean,float(record['mean_plddt_all_atoms_from_cif']),abs_tol=1e-10)
                if i==0:
                    assert (ROOT/f'mapped_structures/FLM_{var}.pdb').read_bytes()==(ROOT/f'converted_pdb/fold_flm_{var}_model_0.pdb').read_bytes()
                    historical=json.loads((ROOT/'historical_annotation/structure_info.json').read_text())[f'FLM_{var}']
                    assert historical['best_match']=='reference_structures/AFDB_Q38836.pdb'
                    assert math.isclose(atom_mean,historical['plddt'],abs_tol=1e-10)
                    assert math.isclose(0.5*atom_mean+50,historical['structure_score'],abs_tol=1e-10)
                model_count+=1
    print(json.dumps(dict(status='passed',files_hash_verified=checked,canonical_CDS_peptide_pairs_verified=2,structure_models_verified=model_count,TM_score_partner='AFDB_Q38836; score alignment not rerun',new_prediction_database_or_language_model_calls=0),indent=2))
if __name__=='__main__':main()
