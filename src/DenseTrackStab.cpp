#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include <time.h>

using namespace cv;

clock_t start, end;
double elapsed_secs;
double total_elapsed_secs;
int count = 0;

std::pair<double,double> find_mean_variance(arma::mat &H, arma::mat &left, arma::mat &right, arma::mat &predict_right, arma::mat &diff) {

    double error = 0;
    double sqsum=0;
    for (int i=0; i<right.n_rows; i++) {
        double k = diff(i,0)*diff(i,0) + diff(i,1)*diff(i,1) + diff(i,2)*diff(i,2);
        error += k;
        sqsum += k*k;
    }
    double mean = error/right.n_rows;
    double variance = sqsum/right.n_rows - mean*mean;

    return std::make_pair(mean, variance);
}

cv::Mat findHomographyOurs(std::vector<cv::Point2f> prev_pts, std::vector<cv::Point2f> pts) {
    arma::mat left = arma::ones(prev_pts.size(), 3);
    arma::mat right = left;
    //printf("%d %d\n", left.n_rows, left.n_cols);
    //printf("%d %d\n", right.n_rows, right.n_cols);
    arma::uword cnt = 0;
    for (auto pt: prev_pts) {
        double x = pt.x;
        double y = pt.y;

        //printf("left %d 0 1\n",cnt);
        left(cnt,0) = x;
        left(cnt,1) = y;
        cnt ++;
    }

    cnt = 0;
    for (auto pt: pts) {
        double x = pt.x;
        double y = pt.y;

        //printf("right %d 0 1\n",cnt);
        right(cnt,0) = x;
        right(cnt,1) = y;
        cnt ++;
    }

    double outlier_ratio;
    arma::mat H;
    int iter = 0;
    do {
        H = arma::pinv(left)*right;

        arma::mat predict_right = left*H;
        arma::mat diff = predict_right - right;
        std::pair<double, double> mean_variance = find_mean_variance(H, left, right, predict_right, diff);
        double mean = mean_variance.first;
        double variance = mean_variance.second;

        //arma::mat new_left = left, new_right = right;
        double threshold = mean + sqrt(variance);
        //printf("Mean: %lf Variance:%lf Threshold:%lf\n", mean, variance, threshold);
        int from_loc = 0, to_loc = 0;
        std::vector< std::pair<double, int> > errors;
        for (int i=0; i<left.n_rows; i++) {
            //double error = diff(i) * diff(i);
            double error = diff(i,0)*diff(i,0) + diff(i,1)*diff(i,1) + diff(i,2)*diff(i,2);
            errors.push_back( std::make_pair(error, i) );
            if (error > threshold)
                to_loc ++;
            else {
                from_loc++;
                to_loc++;
                //new_left(to_loc++) = left(from_loc++);
                //new_right(to_loc++) = right(from_loc++);
            }
        }
        sort(errors.begin(), errors.end());
        int len = errors.size();
        int keep_len = 0.9*len;
        arma::mat new_left(keep_len, 3), new_right(keep_len, 3);
        for (int i=0; i<keep_len; i++) {
            for (int j=0; j<3; j++) {
                new_left(i,j) = left(errors[i].second,j);
                new_right(i,j) = right(errors[i].second,j);
            }
        }

        outlier_ratio = ((double)(from_loc))/to_loc;
        left = new_left;
        right = new_right;
        //printf("Outliers: %d Total: %d Ratio: %lf\n", from_loc, to_loc, outlier_ratio);
        iter++;
    } while (iter < 4);


        //for(int i=0; i<3; i++) {
            //for (int j=0; j<3; j++)
                //printf("%lf ", H(i,j));
            //printf("\n\n");
        //}
    cv::Mat opencv_H(3, 3, CV_64FC1, H.memptr());
    //for(int i=0; i<3; i++)
        //for(int j=0; j<3; j++)
            //opencv_H.at<double>(i,j) = H(i,j);
    return opencv_H;
}

void print_matrix(Mat mat_print) {
    for (int i = 0; i < 3; i++) {
        for (int j=0; j<3; j++)
            printf("%lf ", mat_print.at<double>(i,j));
        printf("\n");
    }
}

void print_time(int line) {
    count++;
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    //printf("Line %d: %d: %lf\n", line, count, elapsed_secs);
    //fflush(stdout);
}

int main(int argc, char** argv)
{
	// IO operation
        start = clock();

	
        print_time(__LINE__);
	const char* keys =
		{
			"{ f  | video_file     | test.avi | filename of video }"
			"{ o  | idt_file   | test.bin | filename of idt features }"
			"{ r  | tra_file   | tra.bin  | filename of track files  }"
			"{ L  | track_length   | 15 | the length of trajectory }"
			"{ S  | start_frame     | 0 | start frame of tracking }"
			"{ E  | end_frame | 1000000 | end frame of tracking }"
			"{ W  | min_distance | 5 | min distance }"
			"{ N  | patch_size   | 32  | patch size }"
			"{ s  | nxy_cell  | 2 | descriptor parameter }"
			"{ t  | nt_cell  | 3 | discriptor parameter }"
			"{ A  | scale_num  | 8 | num of scales }"
			"{ I  | init_gap  | 1 | gap }"
			"{ T  | show_track | 0 | whether show tracks}"
                        "{ h  | use_new_homography_method | false | whether to use our homography method }"
		};
	CommandLineParser cmd(argc, argv, keys);
	string video = cmd.get<string>("video_file");
	string out_file = cmd.get<string>("idt_file");
	string tra_file = cmd.get<string>("tra_file");
	track_length = cmd.get<int>("track_length");
	start_frame = cmd.get<int>("start_frame");
	end_frame = cmd.get<int>("end_frame");
	min_distance = cmd.get<int>("min_distance");
	patch_size = cmd.get<int>("patch_size");
	nxy_cell = cmd.get<int>("nxy_cell");
	nt_cell = cmd.get<int>("nt_cell");
	scale_num = cmd.get<int>("scale_num");
	init_gap = cmd.get<int>("init_gap");
        bool use_new_homography_method = cmd.get<bool>("use_new_homography_method");

	FILE* outfile = fopen(out_file.c_str(), "wb");
	FILE* trafile = fopen(tra_file.c_str(), "wb");

        print_time(__LINE__);
	VideoCapture capture;
	capture.open(video);
	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing %s\n",video.c_str());
		return -1;
	}

        print_time(__LINE__);
	float frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);


        print_time(__LINE__);
	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video.c_str());

	std::vector<Frame> bb_list;
	if(bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	//if(flag)
		 // seqInfo.length = end_frame - start_frame + 1;
    
        print_time(__LINE__);
        printf( "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
        print_time(__LINE__);
	while(true) {
		Mat frame;
		int i, j, c;

		// get a new frame
                print_time(__LINE__);
		capture >> frame;
                print_time(__LINE__);
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

                print_time(__LINE__);
		if(frame_num == start_frame) {
                        print_time(__LINE__);
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);
                        print_time(__LINE__);

			InitPry(frame, fscales, sizes);

                        print_time(__LINE__);
			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

                        print_time(__LINE__);
			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

                        print_time(__LINE__);
			xyScaleTracks.resize(scale_num);

                        print_time(__LINE__);
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

                        print_time(__LINE__);
			for(int iScale = 0; iScale < scale_num; iScale++) {
                                print_time(__LINE__);
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
                                print_time(__LINE__);
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
                                print_time(__LINE__);
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
                        print_time(__LINE__);
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

                        print_time(__LINE__);
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
                        print_time(__LINE__);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		init_counter++;
                print_time(__LINE__);
		frame.copyTo(image);
                print_time(__LINE__);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
                print_time(__LINE__);
		if(bb_file)
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
                print_time(__LINE__);

		// compute optical flow for all scales once
                print_time(__LINE__);
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

                print_time(__LINE__);
		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
                print_time(__LINE__);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
                        start = clock();
                        Mat temp;
                        if (use_new_homography_method)
                            temp = findHomographyOurs(prev_pts_all, pts_all);
                        else
                            temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
                        end = clock();
                        elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
                        total_elapsed_secs += elapsed_secs;
                        printf("Elapsed seconds: %lf\n", elapsed_secs);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

                print_time(__LINE__);
		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
                print_time(__LINE__);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);


                print_time(__LINE__);
		for(int iScale = 0; iScale < scale_num; iScale++) {
                        print_time(__LINE__);
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

                        print_time(__LINE__);
			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

                        print_time(__LINE__);
			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

                        print_time(__LINE__);
			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
                        print_time(__LINE__);
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

				// get the descriptors for the feature point
                                print_time(__LINE__);
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

                                print_time(__LINE__);
				// draw the trajectories at the first scale
				//if(show_track == 1 && iScale == 0)
				//	DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
                                print_time(__LINE__);
				if(iTrack->index >= trackInfo.length) {
        
					std::vector<Point2f> trajectory(trackInfo.length+1), trajectory1(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i){
						trajectory[i] = iTrack->point[i]*fscales[iScale];
						trajectory1[i] = iTrack->point[i]*fscales[iScale];
					}
				
					std::vector<Point2f> displacement(trackInfo.length);
                                        print_time(__LINE__);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
						if(show_track == 1 && iScale == 0)
							DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

						// output the basic information
                                                print_time(__LINE__);
						fwrite(&frame_num,sizeof(frame_num),1,outfile);
						fwrite(&mean_x,sizeof(mean_x),1,outfile);
						fwrite(&mean_y,sizeof(mean_y),1,outfile);
						fwrite(&var_x,sizeof(var_x),1,outfile);
						fwrite(&var_y,sizeof(var_y),1,outfile);
						fwrite(&length,sizeof(var_y),1,outfile);
						float cscale = fscales[iScale];
                                                print_time(__LINE__);
						fwrite(&cscale,sizeof(cscale),1,outfile);

						// for spatio-temporal pyramid
						float temp = std::min<float>(max<float>(mean_x/float(seqInfo.width), 0), 0.999);
						fwrite(&temp,sizeof(temp),1,outfile);
						temp = std::min<float>(max<float>(mean_y/float(seqInfo.height), 0), 0.999);
						fwrite(&temp,sizeof(temp),1,outfile);
						temp =  std::min<float>(max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999);
						fwrite(&temp,sizeof(temp),1,outfile);
					
						// output trajectory point coordinates
                                                print_time(__LINE__);
 				                for (int i=0; i< trackInfo.length; ++ i){
							temp = trajectory1[i].x;
							fwrite(&temp, sizeof(temp), 1, outfile);
							fwrite(&temp, sizeof(temp), 1, trafile);
							temp = trajectory1[i].y;
							fwrite(&temp, sizeof(temp), 1, outfile);
							fwrite(&temp, sizeof(temp), 1, trafile);
                                                        print_time(__LINE__);
						}
              
						// output the trajectory features
						for (int i = 0; i < trackInfo.length; ++i){
							temp = displacement[i].x;
							fwrite(&temp,sizeof(temp),1,outfile);
							temp = displacement[i].y;
							fwrite(&temp,sizeof(temp),1,outfile);
                                                        print_time(__LINE__);
						}

						PrintDesc(iTrack->hog, hogInfo, trackInfo, outfile);
						PrintDesc(iTrack->hof, hofInfo, trackInfo, outfile);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, outfile);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, outfile);
                                                print_time(__LINE__);
					}

					iTrack = tracks.erase(iTrack);
                                        print_time(__LINE__);
					continue;
				}
				++iTrack;
			}
                        print_time(__LINE__);
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);
                        print_time(__LINE__);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
                        print_time(__LINE__);
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
                        print_time(__LINE__);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
                print_time(__LINE__);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);
                print_time(__LINE__);

		frame_num++;
                print_time(__LINE__);

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}


	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");
        print_time(__LINE__);

	fclose(outfile);
	fclose(trafile);
//	fclose(flowx);
//	fclose(flowy);

        printf("Total elapsed_secs: %lf\n", total_elapsed_secs);
	return 0;
}
