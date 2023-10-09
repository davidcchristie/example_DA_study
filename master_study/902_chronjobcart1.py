# ==================================================================================================
# --- Imports
# ==================================================================================================
import tree_maker
import os


# ==================================================================================================
# --- Class for job submission
# ==================================================================================================
class cluster:
    def __init__(self, run_on="local_pc"):
        if run_on in ["local_pc", "htc", "slurm", "htc_docker", "slurm_docker"]:
            self.run_on = run_on
        else:
            import sys

            sys.exit("Error: Submission mode specified is not yet implemented")

    def create_sub_file(self, list_of_nodes, filename="file.sub"):
        running_jobs = self.running_jobs()
        print(f"running: {running_jobs}")
        queuing_jobs = self.queuing_jobs()
        print(f"queuing: {queuing_jobs}")
        l_filename = []
        # Slurm docker is a peculiar case as one submission file must be created per job
        if self.run_on == "slurm_docker":
            for idx_node, node in enumerate(list_of_nodes):
                # Get path node
                path_node = node.get_abs_path()
                if node.has_been("completed"):
                    print(f"{path_node} is completed.")
                elif path_node in running_jobs:
                    print(f"{path_node} is running.")
                elif node in queuing_jobs:
                    print(f"{path_node} is queuing.")
                else:
                    filename_node = filename.split(".sub")[0] + f"_{idx_node}.sub"
                    with open(filename_node, "w") as fid:
                        # Head
                        fid.write("#!/bin/bash\n")
                        fid.write("# This is a SLURM submission file using Docker\n")
                        fid.write("#SBATCH --partition=slurm_hpc_acc\n")
                        fid.write("#SBATCH --output=output.txt\n")
                        fid.write("#SBATCH --error=error.txt\n")
                        fid.write("#SBATCH --ntasks=2\n")

                        # Careful, I implemented a fix for path due to the temporary home recovery folder
                        to_replace = "/storage-hpc/gpfs_data/HPC/home_recovery"
                        replacement = "/home/HPC"
                        fixed_path = path_node.replace(to_replace, replacement)
                        # update path for sed
                        to_replace = to_replace.replace("/", "\/")
                        replacement = replacement.replace("/", "\/")
                        # Mutate path in run.sh and other potentially problematic files
                        fid.write(f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/run.sh\n")
                        fid.write(
                            f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/config.yaml\n"
                        )
                        # Final submission line
                        fid.write(
                            f"singularity exec ../da-study-docker_latest.sif {fixed_path}/run.sh\n"
                        )

                        # Tail
                        fid.write(f"#{self.run_on}\n")
                    l_filename.append(filename_node)

        # Htcondor, local, etc.
        else:
            with open(filename, "w") as fid:
                # head
                if self.run_on == "local_pc":
                    fid.write("# Running on local pc\n")
                elif self.run_on == "lsf":
                    fid.write("# Running on LSF \n")
                elif self.run_on == "slurm":
                    fid.write("# Running on SLURM \n")
                elif self.run_on == "htc":
                    fid.write("# This is a HTCondor submission file\n")
                    fid.write("error  = error.txt\n")
                    fid.write("output = output.txt\n")
                    fid.write("log  = log.txt\n")

                elif self.run_on == "htc_docker":
                    fid.write("# This is a HTCondor submission file using Docker\n")
                    fid.write("error  = error.txt\n")
                    fid.write("output = output.txt\n")
                    fid.write("log  = log.txt\n")
                    fid.write("universe = vanilla\n")
                    fid.write(
                        "+SingularityImage ="
                        ' "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker:latest"\n'
                    )
                    # fid.write("transfer_output_files=config.yaml\n")

                    # if user has defined a htc_job_flavor in config.yaml otherwise default is "espresso"
                    if (
                        "htc_job_flavor"
                        in list_of_nodes[0].root.parameters["generations"][
                            str(list_of_nodes[0].depth)
                        ]
                    ):
                        htc_job_flavor = list_of_nodes[0].root.parameters["generations"][
                            str(list_of_nodes[0].depth)
                        ]["htc_job_flavor"]
                        fid.write(f'+JobFlavour  = "{htc_job_flavor}"\n')

                for idx_node, node in enumerate(list_of_nodes):
                    # Get path node
                    path_node = node.get_abs_path()
                    if node.has_been("completed"):
                        print(f"{path_node} is completed.")
                    elif path_node in running_jobs:
                        print(f"{path_node} is running.")
                    elif node in queuing_jobs:
                        print(f"{path_node} is queuing.")
                    else:
                        if self.run_on == "local_pc":
                            fid.write("bash " + path_node + "/run.sh &\n")
                        elif self.run_on == "lsf":
                            fid.write(
                                "bsub -n 2 -q hpc_acc "
                                # "-e error.txt -o output.txt "
                                f"{path_node}/run.sh\n"
                            )
                        elif self.run_on == "slurm":
                            fid.write(
                                "sbatch --ntasks=2 --partition=slurm_hpc_acc --output=output.txt "
                                f"{path_node}/run.sh\n"
                            )

                        elif self.run_on == "htc" or self.run_on == "htc_docker":
                            # initialdir is needed so that each job has it own output, error and log.txt
                            fid.write(f"initialdir = {path_node}\n")
                            fid.write(f"executable = {path_node}/run.sh\nqueue\n")

                # tail
                if (
                    self.run_on == "local_pc"
                    or self.run_on == "lsf"
                    or self.run_on == "slurm"
                    or self.run_on == "htc"
                    or self.run_on == "htc_docker"
                ):
                    fid.write(f"#{self.run_on}\n")
            l_filename.append(filename)
        return l_filename

    def submit(self, l_filename):
        for filename in l_filename:
            if self.run_on == "local_pc":
                print(f"bash {filename}")
                os.system(f"bash {filename}")
            elif self.run_on == "lsf":
                os.system(f"bash {filename}")
            elif self.run_on == "slurm":
                os.system(f"bash {filename}")
            elif self.run_on == "slurm_docker":
                os.system(f"sbatch {filename}")
            elif self.run_on == "htc" or self.run_on == "htc_docker":
                os.system(f"condor_submit {filename}")

    def running_jobs(self):
        my_list = []
        if self.run_on == "local_pc":
            # Does not work at the moment in lxplus
            # for ps in psutil.pids():
            #    aux=psutil.Process(ps).cmdline()
            #    if len(aux)>1:
            #        if 'run.sh' in aux[-1]:
            #            my_list.append(str(Path(psutil
            #                                    .Process(ps)
            #                                    .cmdline()[-1])
            #                               .parent))
            return []

            return my_list
        elif self.run_on == "lsf":
            return []
        elif self.run_on == "slurm" or self.run_on == "slurm_docker":
            return []
        elif self.run_on == "htc" or self.run_on == "htc_docker":
            return []

    def queuing_jobs(self):
        return []


# ==================================================================================================
# --- Submission
# ==================================================================================================
# Load the tree from a yaml
if __name__ == "__main__":
    study_name = "cart1"
    fix = "/scans/" + study_name
    root = tree_maker.tree_from_json(fix[1:] + "/tree_maker_" + study_name + ".json")
    # Add suffix to the root node path to handle scans that are not in the root directory
    root.add_suffix(suffix=fix)

    if root.has_been("completed"):
        print("All descendants of root are completed!")
    else:
        my_run_on = cluster(root.parameters["generations"]["1"]["run_on"])
        my_file = "submission_files/first_generation.sub"
        l_filename = my_run_on.create_sub_file(root.generation(1), my_file)
        my_run_on.submit(l_filename)
        if all([node.has_been("completed") for node in root.generation(1)]):
            my_run_on = cluster(root.parameters["generations"]["2"]["run_on"])
            my_file = "submission_files/second_generation.sub"
            l_filename = my_run_on.create_sub_file(root.generation(2), my_file)
            my_run_on.submit(l_filename)
        if all([descendant.has_been("completed") for descendant in root.descendants]):
            root.tag_as("completed")
            print("All descendants of root are completed!")
#        else:
#            for descendant in root.descendants:
#                if descendant.has_not_been('completed'):
#                     print("To be completed: "+ descendant.get_abs_path())
