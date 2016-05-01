import sys
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plot
# import pandas
import pandas as pd
import scipy
import sklearn.cluster as skc
from sklearn.cluster import DBSCAN

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import distance
from operator import itemgetter

import datetime

def FLV(pathIn, pathOut):
	print 'program running'
	if pathIn == '' or pathOut == '':
		mFile = open("D:\\Projects\\aag_2016\\chicago_2014_final_traj.txt","rb")
		mResultFile = open("top_locations.txt", "wb")
	else:
		mFile = open(pathIn, "rb")
		mResultFile = open(pathOut, "wb")

	for indx, mRow in enumerate(mFile):
		print 'processing ...' + str(indx)
		tweets_locations = []
		mTime = []
		mLandUseC = []
		mPolyIDC = []
		mTraj = mRow.replace('\n', '').split(',')
		
		for mLoc in mTraj[1:]:
			xx = []
			[lng, lat, mTimeStamp, mPolyID, mLandUse] = mLoc.split('&')
			xx.append(float(lng))
			xx.append(float(lat))
			tweets_locations.append([float(lng), float(lat), mTimeStamp, mPolyID, mLandUse])

		if len(tweets_locations) > 4:
			myarray = np.asarray(tweets_locations)
			df = pd.DataFrame(myarray)
			df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
			geo_positions = df[['lng', 'lat']]

			# print df
			# print len(tweets_locations)
			# # print myarray
			# # D = distance.squareform(distance.pdist(tweets_locations))
			# # S = 1 - (D / np.max(D))
			
			EPSILON = 0.0025 # this is NOT 500 meters, those are flat 0.005 degrees, which are ~555 meters (simpler)
			SAMPLES = 1  # this is the number of minimum samples (people), ten yields too many cluster, 50 seems good
			db  = skc.DBSCAN(eps=EPSILON, min_samples=SAMPLES)
			labels  = db.fit_predict(geo_positions.values)
			core_samples = db.core_sample_indices_
			labels = db.labels_
			# print labels
			df['cluster'] = labels
			# print df
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			# print n_clusters_

			grouped = df.groupby(['cluster'])

			result_temp = []

			for key, group in grouped:
				if key != -1.0:
					result_temp.append([key, group['cluster'].count()])

			if result_temp != []:
				# m = np.array(result_temp)
				# sort_rank = np.argsort(m, axis=0)
				# for m in sort_rank:
				result = sorted(result_temp, key=itemgetter(1), reverse=True)
				# print result
				for indx, m in enumerate(result):
					df.loc[df['cluster'] == m[0], 'rank'] = indx
			print df






	# 			mOutput = ''
	# 			for mComponents in result:
	# 				mOutput += mComponents[2] + ":" + str(mComponents[3]) + ";" 
	# 			mResultFile.write(mOutput)
	# 			mResultFile.write("\n")
	
	# mResultFile.close()


from collections import Counter
def get_top_locations_from_cluster(cluster):
	words_to_count = (word for word in cluster if word)
	c = Counter(words_to_count)
	mMostCount = c.most_common()[:1]
	return [mMostCount[0][0], (mMostCount[0][1]*1.0) /(len(cluster) * 1.0)]

def get_top_location_from_cluster(cluster):
	words_to_count = (word for word in cluster if word)
	c = Counter(words_to_count)
	mMostCount = c.most_common()[:1]
	return [mMostCount[0][0], (mMostCount[0][1]*1.0) /(len(cluster) * 1.0), cluster]



def cluster_summary():
	openFileName = askopenfilename()
	mFile = open(openFileName, "rb")
	mOutputFVL = open("result/number_of_fvl.txt", "wb")

	# #data format: 1100:0.754385964912;1215:0.363636363636;1100:0.941176470588;
	
	for row in mFile:
		print '----------------'
		unique_landuse = []
		record = row.replace(";\n", "").split(";")
		# print record
		total_num_cluster = len(record)
		mOutputFVL.write(str(total_num_cluster) + '\n')

		for mm in record:
			[landuse, purity] = mm.split(":")
			unique_landuse.append(landuse)
		uniq_landuse, counts = np.unique(unique_landuse, return_counts=True)
		# print uniq_landuse
		
		temp_result = np.asarray((uniq_landuse, counts)).T
		print temp_result

		for uniq in temp_result:
			with open("result/landuse_" + uniq[0] + ".csv", "a") as myfile:
				myfile.write(str(total_num_cluster) + ',' + uniq[1] + '\n')
		myfile.close()

	mOutputFVL.close()


def cluster_summary2():
	openFileName = askopenfilename()
	mFile = open(openFileName, "rb")
	mOutputFVL = open("result/number_of_fvl.txt", "wb")

	# #data format: 1100:0.754385964912;1215:0.363636363636;1100:0.941176470588;
	
	for row in mFile:
		print '----------------'
		unique_landuse = []
		record = row.replace(";\n", "").split(";")
		# print record
		total_num_cluster = len(record)
		mOutputFVL.write(str(total_num_cluster) + '\n')

		for mm in record:
			[landuse, purity] = mm.split(":")
			unique_landuse.append(landuse)
		uniq_landuse, counts = np.unique(unique_landuse, return_counts=True)
		# print uniq_landuse
		
		temp_result = np.asarray((uniq_landuse, counts)).T
		print temp_result

		# for uniq in temp_result:
		# 	with open("result/landuse_" + uniq[0] + ".csv", "a") as myfile:
		# 		myfile.write(str(total_num_cluster) + ',' + uniq[1] + '\n')
		# myfile.close()

	mOutputFVL.close()



def tweets_with_dates():
	print 'coverting tweets with dates'
	openFileName = askopenfilename()
	mFile = open("/home/jun/project/chicago/chicago_2014_final_traj.txt", "rb")
	# mFile = open("chicago_2014_final_traj.txt","rb")

	for mRow in mFile:
		tweets_locations = []
		mTime = []
		mLandUseC = []
		mPolyIDC = []
		mTraj = mRow.replace('\n', '').split(',')
		
		for mLoc in mTraj[1:]:
			xx = []
			[lng, lat, mTimeStamp, mPolyID, mLandUse] = mLoc.split('&')
			xx.append(float(lng))
			xx.append(float(lat))
			tweets_locations.append([float(lng), float(lat), int(mTimeStamp), mPolyID, mLandUse])

		
		myarray = np.asarray(tweets_locations)
		df = pd.DataFrame(myarray)
		df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
		geo_positions = df[['lng', 'lat']]
		df['timestamp'] = df['timestamp'].astype(int)
		for ff in df['timestamp']:
			xx = datetime.datetime.fromtimestamp(ff/1000).strftime('%m-%d')
			if xx > '03-20' and xx < '11-00':
				# print 'cst'
				df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 5
			else:
				df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 6

		df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

		for tt in df['timestamp']:
			df.loc[df['timestamp'] == tt, 'hour'] = tt.hour
			df.loc[df['timestamp'] == tt, 'day_of_week'] = tt.isoweekday()
		with open("/home/jun/project/chicago/result/chicago_2014_all_dates.csv", "a") as myfile:
				df.to_csv(myfile, sep='\t', mode='a', header=False, index=False)
				
def tweets_with_dates_cluster_info():
	print 'coverting tweets with dates'
	mFile = open("D:\\Projects\\aag_2016\\chicago_2014_final_traj.txt", "rb")

	for indx, mRow in enumerate(mFile):
		print 'processing ...' + str(indx)
		tweets_locations = []
		mTime = []
		mLandUseC = []
		mPolyIDC = []
		mTraj = mRow.replace('\n', '').split(',')

		uid = mTraj[0]
		
		for mLoc in mTraj[1:]:
			xx = []
			[lng, lat, mTimeStamp, mPolyID, mLandUse] = mLoc.split('&')
			xx.append(float(lng))
			xx.append(float(lat))
			tweets_locations.append([float(lng), float(lat), mTimeStamp, mPolyID, mLandUse])

		if len(tweets_locations) > 35:
		# if len(tweets_locations) > 4 and len(tweets_locations) < 2683:
			myarray = np.asarray(tweets_locations)
			df = pd.DataFrame(myarray)
			df.columns = ['lng', 'lat', 'timestamp', 'polyID', 'landuse']
			geo_positions = df[['lng', 'lat']]
			
			EPSILON = 0.0025 # this is NOT 500 meters, those are flat 0.005 degrees, which are ~555 meters (simpler)
			SAMPLES = 1  # this is the number of minimum samples (people), ten yields too many cluster, 50 seems good
			db  = skc.DBSCAN(eps=EPSILON, min_samples=SAMPLES)
			labels  = db.fit_predict(geo_positions.values)
			core_samples = db.core_sample_indices_
			labels = db.labels_
			# print labels
			df['cluster'] = labels
			df['uid'] = uid 

			grouped = df.groupby(['cluster'])

			result_temp = []

			for key, group in grouped:
				if key != -1.0:
					result_temp.append([key, group['cluster'].count()])

			if result_temp != []:
				# m = np.array(result_temp)
				# sort_rank = np.argsort(m, axis=0)
				# for m in sort_rank:
				result = sorted(result_temp, key=itemgetter(1), reverse=True)
				# print result
				for indx, m in enumerate(result):
					df.loc[df['cluster'] == m[0], 'rank'] = indx

			df['timestamp'] = df['timestamp'].astype(int)
			
			for ff in df['timestamp']:
				xx = datetime.datetime.fromtimestamp(ff/1000).strftime('%m-%d')
				if xx > '03-20' and xx < '11-00':
					# print 'cst'
					df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 5
				else:
					df.loc[df['timestamp'] == ff, 'timestamp'] = ff/1000 - 3600 * 6

			df['timestamp2'] = pd.to_datetime(df['timestamp'], unit='s')

			for tt in df['timestamp2']:
				df.loc[df['timestamp2'] == tt, 'hour'] = tt.hour
				df.loc[df['timestamp2'] == tt, 'day_of_week'] = tt.isoweekday()
			with open("/home/jun/project/chicago/result/chicago_2014_rank.csv", "a") as myfile:
				df.to_csv(myfile, sep='\t', mode='a', header=False, index=False)


if __name__ == '__main__':
	# FLV('','')
	tweets_with_dates_cluster_info()
