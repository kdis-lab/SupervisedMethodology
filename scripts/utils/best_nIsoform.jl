
using CSV

path = "/home/eperezp1990/cbms/fase final/3/results"
pathReport = "/home/eperezp1990/cbms/fase final/resultsBest.csv"
files = [[joinpath(root,f) for f in files if occursin("/all", root)] for (root, dirs, files) in walkdir(path) if length(files) != 0]
files = [i for i in files if size(i) != (0,)]

open(pathReport, "w") do file
	write(file, join(["DATASET", "attrs", "auc"], ","),"\n")
	for dataset in files
		for f in dataset
			df = CSV.read(f)
			write(file,join([f, df[Symbol("NumberAtts")][1], df[Symbol("metricOpt")][1]], ","), "\n")
		end
	end
end


