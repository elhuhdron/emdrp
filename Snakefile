#snakemake -j --use-conda # local execution
#snakemake --profile axon -j --use-conda --use-envmodules # slurm exectution on axon

configfile: "config.yml"
root = config['local_data_path']


# Workaround for rest2skel; Better: add all shell scripts to PATH environment variable via setup tools ...
if 'emdrp' in workflow.modules.keys():
    cwd = str(Path(workflow.modules['emdrp'].snakefile).parent) 
else:
    cwd = '.'

from pathlib import Path

localrules:
    merge_predicted_probabilities,
    produce_metrics,
    plot_metrics,
    store_volume_in_correct_location,


rule all:
    input:
        #expand(root + '/data_out/tutorial_ECS/xfold/M0007_plots/1000.fig',
        #    ident=['M0007', 'M0027']),
        root + '/data_vols/M0007_random_test.h5',
        root + '/data_vols/M0007_original_test.h5',


rule store_volume_in_correct_location:
    output:
        root + '/data_vols/{ident}_{type}_{extra}.h5'
    input:
        root + '/data_restored/{ident}_{type}_{extra}.h5'
    params:
        original_volume = lambda wc: config['datasets'][wc.ident]['original_volume'],
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
        size = lambda wc: config['datasets'][wc.ident]['size'],
        dataset_name = 'data_mag1',
    conda:
        'environment.yml'
    shell:
         f"{cwd}/recon/emdrp/dpWriteh5.py" +
         " --srcfile {params.original_volume} --outfile {output} " +
         "--chunk {params.chunk} --size {params.size} " +
         "--offset 0 0 0 --dataset {params.dataset_name} --inraw {input} --dpW"


rule merge_predicted_probabilities:
    output:
        root + '/data_out/{ident}_{type}_{extra}_probs.h5',
    input:
        expand(root + '/data_out/{{ident}}_{{type}}_{{extra}}_{replicate}.0_probs.h5',
            replicate =range(4)),
    wildcard_constraints:
        ident="(M0007)|(M0027)",
        type="[^_]+",
        extra="[^_]+",
    params:
        src_path = root + '/data_out/',
        srcfiles = lambda wc: [p.name for p in Path(root + '/data_out/').glob(f'{wc.ident}_{wc.type}_{wc.extra}_*.0_probs.h5')],
        dim_order = lambda wc: ['xyz' for p in Path(root + '/data_out/').glob(f'{wc.ident}_{wc.type}_{wc.extra}_*.0_probs.h5')],
        size = lambda wc: config['datasets'][wc.ident]['size'],
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    conda:
        'environment.yml'
    shell:
        f'python -u {cwd}/recon/emdrp/dpMergeProbs.py' +
        ' --srcpath {params.src_path}' +
        ' --srcfiles {params.srcfiles}' +
        ' --dim-orderings {params.dim_order}' +
        ' --outprobs {output}' +
        ' --chunk {params.chunk}' +
        ' --size {params.size}' +
        ' --types ICS ECS MEM --ops mean min --dpMergeProbs-verbose'

rule apply_watershed_on_ICS_probability:
    output:
         root + '/data_out/{ident}_{type}_{extra}_supervoxels.h5',
    input:
        root + '/data_out/{ident}_{type}_{extra}_probs.h5'
    params:
        size = lambda wc: config['datasets'][wc.ident]['size'],
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    resources:
        time='12:00:00',
        partition="p.gpu", # since cpu queue is full
        mem="64000",
        cpus_per_task="2",
    conda:
        'environment.yml'
    shell:
        f'python -u {cwd}/recon/emdrp/dpWatershedTypes.py' +
        ' --probfile {input}' +
        ' --chunk {params.chunk} --offset 0 0 0 --size {params.size}' +
        ' --outlabels {output}' +
        ' --ThrRng 0.5 0.999 0.1' +
        ' --ThrHi 0.95 0.99 0.995 0.999 0.99925 0.9995 0.99975 0.9999 0.99995 0.99999 --dpW'

rule produce_metrics:
    output:
       root + '/data_out/{ident}_{type}_{extra}_output.mat'
    input:
        h5_raw_data_path = lambda wc: config['datasets'][wc.ident]['original_volume'],
        lblsh5 = root + '/data_out/{ident}_{type}_{extra}_supervoxels.h5',
        skelin = lambda wc: config['datasets'][wc.ident]['skeleton'],
    params:
        chunk = lambda wc: config['datasets'][wc.ident]['chunk'],
    envmodules:
        'matlab/R2020b'
    shell:
        f"""matlab -nojvm -nosplash -batch "addpath(genpath('{cwd}/recon/matlab'));""" + """ knossos_efpl_top_snakemake('{output}', '{input.lblsh5}', '{input.h5_raw_data_path}', '{input.skelin}', [{params.chunk}])" """

rule plot_metrics:
    output:
        fig1000 = root + '/data_out/{ident}_{type}_{extra}_plots/1000.fig',
        fig1001 = root + '/data_out/{ident}_{type}_{extra}_plots/1001.fig',
        fig1002 = root + '/data_out/{ident}_{type}_{extra}_plots/1002.fig',
        fig1003 = root + '/data_out/{ident}_{type}_{extra}_plots/1003.fig',
    input:
        input_mat = root + '/data_out/{ident}_{type}_{extra}_output.mat'
    params:
        output_path = lambda wc: root + f'/data_out/{wc.ident}_{wc.type}_{wc.extra}_plots',
    envmodules:
        'matlab/R2020b'
    shell:
        f"""matlab -nosplash -batch "addpath(genpath('{cwd}/recon/matlab'));""" + """ knossos_efpl_plot_top_snakemake('{params.output_path}', '{input.input_mat}')" """