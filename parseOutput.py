import re

# use regex to filter lines and capture values
p1 = re.compile(r"P = (?P<processes>\d*), t = (?P<threads>\d*)")
p4 = re.compile(r"t = (?P<threads>\d*), k = (?P<exponent>\d*)")
p5 = re.compile(r"P = (?P<processes>\d*), k = (?P<exponent>\d*)")
p2 = re.compile(r"Time elapsed: (?P<time>\d.\d*e[+-]\d*)")
p3 = re.compile(r"  (?P<h>\d*.\d*)[ ]*(?P<error>\d*.\d*)")

total_time = 0

exercises = []
series = []

# helper function to clear a list when new exercise or run is done
def appendToAndClear(bigList, smallList):
	if smallList != []:
		bigList.append(smallList)
		smallList = []

	return bigList, smallList


# Pull out relevant data from output file
with open("output2.txt", "r") as f:
	serie = []
	exercise = []
	for line in f:

		if "Exercise" in line:
			exercise, serie = appendToAndClear(exercise, serie)
			exercises, exercise = appendToAndClear(exercises, exercise)
			exercise = exercise + [line[:-1]]

		elif "t = " in line and " k = " in line:	# Exercise 2 convergence 
			w = p4.match(line)
			exercise, serie = appendToAndClear(exercise, serie)
			serie = serie + [int(w.group("threads")), int(w.group("exponent"))]

		elif "P = " in line and "t = " in line:		# Exercise 2 and 3
			w = p1.match(line)
			exercise, serie = appendToAndClear(exercise, serie)
			serie = serie + [int(w.group("processes")), int(w.group("threads"))]

		elif "P = " in line and "k = " in line:		# Exercise 4
			w = p5.match(line)
			exercise, serie = appendToAndClear(exercise, serie)
			serie = serie + [int(w.group("processes")),int(w.group("exponent"))]

		elif "Time elapsed: " in line:				# Timings for all exercises
			w = p2.match(line)
			serie = serie + [float(w.group("time"))]
			total_time += float(w.group("time"))

		elif "." in line and "00" in line:			# Exercise 2 convergence
			w = p3.match(line)
			serie = serie + [float(w.group("h")), float(w.group("error"))]

print "Total time: ", total_time, "seconds,", total_time/60.0, "min"

# Print til filer
for exercise in exercises:
	if "convergence" in exercise[0]:
		for t in [4,8]:
			with open("convergence_plot_t{}.dat".format(t), "w") as f:
				f.write("h maxRelativeError\n")
				for serie in exercise[1:]:
					if serie[0] == t:
						f.write("{:18.16f} {:19.16f}\n".format(serie[3], serie[4]))

	elif "Exercise 2" in exercise[0]:
		for P in [1,2,4,8,16,32]:
			with open("exercise2_k12_P{}.dat".format(P), "w") as f:
				f.write("t time\n")
				for serie in exercise[1:]:
					if serie[0] == P:
						f.write("{:<2d} {:.3f}\n".format(serie[1], serie[2]))

		for t in [1,2,4,8]:
			with open("exercise2_k12_t{}.dat".format(t), "w") as f:
				f.write("P time\n")
				for serie in exercise[1:]:
					if serie[1] == t:
						f.write("{:<2d} {:.3f}\n".format(serie[0], serie[2]))

	elif "Exercise 3" in exercise[0]:
		with open("exercise3_k14.dat", "w") as f:
			f.write("P time\n")
			for serie in exercise[1:]:
				f.write("{:<2d} {:.3f}\n".format(serie[0], serie[2]))

	elif "Exercise 4" in exercise[0]:
		for P in [1,2,4,8,16,32]:
			with open("exercise4_P{}.dat".format(P), "w") as f:
				f.write("N time\n")
				for serie in exercise[1:]:
					if serie[0] == P:
						f.write("{:<7d} {:.3f}\n".format(2**(2*serie[1]), serie[2]))

		for k in [10,11,12,13,14]:
			with open("exercise4_S_P_k{}.dat".format(k), "w") as f:
				f.write("P S_P\n")
				for serie in exercise[1:]:
					if serie[1] == k:
						f.write("{:<2d} {:.3f}\n".format(serie[0], exercise[1+(k-10)][2]/serie[2]))

		for k in [10,11,12,13,14]:
			with open("exercise4_eta_P_k{}.dat".format(k), "w") as f:
				f.write("P eta_P\n")
				for serie in exercise[1:]:
					if serie[1] == k:
						f.write("{:<2d} {:.3f}\n".format(serie[0], exercise[1+(k-10)][2]/serie[2]/serie[0]))

	for serie in exercise:
		print serie
	print