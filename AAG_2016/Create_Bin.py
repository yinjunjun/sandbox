import numpy as np

def main():
	N = 100
	xmin = 1.0

	#winter 2.2435e+06
	#summer 2.2521e+06
	#spring 2.2325e+06
	#autum 2.2659e+06
	#xmax = 2.2659e+06

	# xmax = 1.3979e+10
	# xmax = 1.1614e+10
	# xmax = 6.8986e+09
	# xmax = 1.1174e+04

	xmax = 6818

	a = np.power(xmax/xmin, 1.0/100)

	mSeies = []
	for i in range(N + 1):
		ww = xmin * np.power(a, i)
		mSeies.append(ww)

	print mSeies
	# print len(mSeies)


if __name__ == '__main__':
	main()