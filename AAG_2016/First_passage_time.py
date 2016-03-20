def main():
	mFile = open("D:\\Projects\\aag_2016\\chicago_2014_final_traj.txt","rb")
	
	for mLine in mFile:
		line = mLine.strip('\n').split(',')
		mNumber = len(line)




if __name__ == '__main__':
	main()