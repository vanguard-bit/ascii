#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/saturate.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include "chars.h"
#define SEC_SZ 8

using namespace cv;
using namespace std;

class parallelConvolution : public ParallelLoopBody {
  private:
    Mat &m_src;
    Mat &m_dst;
    Mat m_kernel;
    int size;
    int channels;
    int rows, cols;
    vector<vector<vector<char>>> &ascii_art;
    int dirs[9][2] = {{0, 0}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}};

  public:
    parallelConvolution(Mat& src, Mat &dst, Mat kernel, vector<vector<vector<char>>> &a)
      : m_src(src), m_dst(dst), m_kernel(kernel), ascii_art(a){
        assert(kernel.rows == kernel.cols && kernel.cols== 3);
        // If using a large kernel create another method that iterates -half of rows to +half of rows
        // cout << m_src;
        size = kernel.rows;
        update_dirs(m_src);
      }
    void update_dirs(Mat m_src){
        channels = m_src.channels();
        rows = m_src.rows;
        cols = m_src.cols;
        // for(int i = 0; i < 9; i++){
        //   dirs[i][0] *= channels;
        //   dirs[i][1] *= channels;
        // }
        // for(auto x: dirs)
        //   cout << x[0] << " " << x[1] << endl;
        // cout << endl;
    }
    auto convolute(Mat msrc, int i, int j, int limiti, int limitj) const{
        double value = 0;
        int count = 0;
        for(auto dxdy: dirs){
          int dx = dxdy[0], dy = dxdy[1];
          if(i + dx >= limiti || i + dx < 0 || j + (dy * channels) >= limitj * channels || j + (dy * channels) < 0)
            continue;
          count++;
          value += (m_kernel.at<double>(1 + dx, 1 + dy) * (msrc.at<uchar>(i + dx, j + (dy * channels))));
        }
        return static_cast<double>(value * (9) / count);
    }

    double convolute_d(Mat msrc, int i, int j, int limiti, int limitj) const{
        double value = 0;
        int count = 0;
        int sz = size / 2;
        for(int dx = -sz; dx <= sz; ++dx){
          if(i + dx >= limiti || i + dx < 0)
            continue;
          // if(1 + dx >= m_kernel.rows || 1 + dx < 0)
          //   continue;
          // cout << "  "   << sz << "     " << "idx = " << 1 + dx << endl;
          // cout << m_kernel << endl;
          const double *ker_ptr = m_kernel.ptr<double>(1 + dx);
          double *src_ptr = msrc.ptr<double>(i + dx);
          // cout << "not this";
          for(int dy = -sz; dy <= sz; ++dy){
            if(
                j + (dy * channels) >= limitj || j + (dy * channels) < 0
              )
                continue;
            count++;
            value += static_cast<double>(ker_ptr[1 + dy] * (msrc.at<double>(i + dx, j + (dy * channels))));
            // cout << "real" << static_cast<double>(ker_ptr[1 + dy] * (msrc.at<double>(i + dx, j + (dy * channels)))) << endl; 

          }
        }
        // cout << value << endl;
        // int _count = count, _value = value; 
        // count = 0;
        // value = 0;


        // for(auto dxdy: dirs){
        //   int dx = dxdy[0], dy = dxdy[1];
        //   if(i + dx >= limiti || i + dx < 0 || j + (dy * channels) >= limitj * channels || j + (dy * channels) < 0)
        //     continue;
        //   count++;
        //   value += (m_kernel.at<double>(1 + dx, 1 + dy) * (msrc.at<double>(i + dx, j + (dy * channels))));
        // }
        // assert(value == _value);
        // assert(count == _count);
        return static_cast<double>(value * (9) / count);
    }

    virtual void operator()(const Range &range) const CV_OVERRIDE {
      // cout << "working" << endl;
      for (int r = range.start; r < range.end; r++) {

        // calculate current postion(i, j) and return if not possible to create roi
        int i = r / (m_src.cols * channels), j = r % (m_src.cols * channels);
        if(i + 4 > rows || j + (4 * channels) > cols * channels)
          continue;

        // create roi strating from current postion (i, j)
        Mat roi(4, 4, CV_8U, Scalar(0));
        uchar max1 = 0, max2 = 0, max3 = 0;
        int max_arr[3][256] = {0, 0, 0};
        int max_val;
        for(int x = 0; x < 4; x++){
          uchar *roi_ptr = roi.ptr(x);
          int *src_ptr = m_src.ptr<int>(i + x);
          for(int y = 0; y < 4; y++){
            int val = src_ptr[j + (channels * y)];
            uchar val1, val2, val3;
            val3 = val & ((1 << 8) - 1);
            val2 = (val >> 8) & ((1 << 8) - 1);
            val1 = (val >> 16) & ((1 << 8) - 1);
            roi_ptr[y] = saturate_cast<uchar>((val1 + val2  + val3) / 3);
            val1 /= 8;
            val2 /= 8;
            val3 /= 8;
            max_arr[0][val1] += 1;
            max_arr[1][val2] += 1;
            max_arr[2][val3] += 1;
          }
        }
        int max_ele[3];
        for(int coli = 0; coli < 3; coli++){
          int most_reps = 0;
          int index = 0;
          for(int colj = 0; colj < 255; colj++){
            // cout << max_arr[coli][colj] << endl;
            if(max_arr[coli][colj] > most_reps){
                most_reps = max_arr[coli][colj];
                index = colj;
            }
          }
          // cout << index << endl;
          max_ele[coli] = (index * 8);
        }
        max_val = (max_ele[0] << 16) + (max_ele[1] << 8) + max_ele[2];
        // cout << roi << endl;


        //check each char out of 95 chars for best match
        double min_val = DBL_MAX;
        int min_index = -1;
        for(int chari = 0; chari < 95; chari++){
          
          // combining currect 8 x 8 image with char matrix
          Mat curr_mat(4, 4, CV_64F, Scalar(0.0));
          for(int x = 0; x < 4; x++){
            uchar *roi_ptr = roi.ptr(x);
            double *curr_ptr = curr_mat.ptr<double>(x);
            for(int y = 0; y < 4; y++){
              curr_ptr[y] = saturate_cast<double>(
                  abs(static_cast<int>(char_matrices[chari][x][y] - roi_ptr[y])));
            }
          }
          // cout << curr_mat << endl;
          // convolute each pixel and sum up the value
          double total_val = 0;
          for(int convi = 0; convi < 4; convi++){
            for(int convj = 0; convj < 4; convj++){
              total_val += convolute_d(curr_mat, 0, 0, 4, 4);
              // cout << convolute_d(curr_mat, 0, 0, 8, 8) << " ";
            }
            // cout << endl;
          }
          total_val /= 64;
          // cout << total_val << " " << min_val << " " << chari << endl;
          // take the minium along all the chars
          if(total_val < min_val){
            min_val = total_val;
            min_index = chari + 32;
          }
          // cout << total_val << " " << min_val << " " << min_index << endl;
          // return;



        }

        //update the best match char in ascii_art vector

        // cout << endl << min_val << " " << min_index << " " << static_cast<char>(min_index) << endl;
        // cout << roi << endl;
        // cout << j % 3 << " " << i << " " << j << endl;
        if(
            j % channels > -1 && j % channels < ascii_art.size() &&
            i > -1 && i < ascii_art[j % channels].size() &&
            j > -1 && j < ascii_art[j % channels][i].size()
          ) {
          // cout << "working";
          ascii_art[j % channels][i][j] = static_cast<char>(min_index);
        }
        m_dst.at<int>(i, j) = saturate_cast<int>(max_val); 
        // cout << "here maybe" << endl;
      }
    }
    friend ostream& operator<<(ostream& out, parallelConvolution obj){
      out << obj.m_dst;
      return out;
    }
};

int main(int argc, char** argv ) {

    Mat image, new_image;
    image = imread("C:\\Users\\Prajwal\\OneDrive\\Desktop\\new\\Cool00086400.jpg", IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::resize(image, image, cv::Size(240, 60));
    int top = (int) (10); 
    int bottom = top;
    int left = (int) (10); 
    int right = left;
    Mat color_image(image.rows, image.cols, CV_32S);
    for(int i = 0; i < image.rows; i++){
      for(int j = 0; j < image.cols; j++){
        uchar val1, val2, val3;
        int val;
        val1 = image.at<uchar>(i, (j * image.channels()));
        val2 = image.at<uchar>(i, (j * image.channels()) + 1);
        val3 = image.at<uchar>(i, (j * image.channels()) + 2);
        val = (val1 << 16) + (val2 << 8) + val3;
        color_image.at<int>(i, j) = val;
      }
    }
    copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(255));
    // cvtColor(image, image, COLOR_BGR2GRAY);
    new_image = Mat(color_image.size(), color_image.type(), Scalar::all(0));
    Mat mat = (Mat_<uchar>(8, 8) << 
        1, 1, 0, 0, 1, 1, 0, 0, 
        1, 1, 1, 0, 1, 1, 1, 0, 
        0, 1, 1, 0, 0, 1, 1, 0, 
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 
        1, 1, 1, 0, 1, 1, 1, 0, 
        0, 1, 1, 0, 0, 1, 1, 0, 
        0, 0, 0, 0, 0, 0, 0, 0
        );
    Mat des(4, 4, CV_8U, Scalar(0));
    double arr[3][3] = {
      {3.0 / 64, 8.0 / 64, 3.0 / 64},
      { 8.0 / 64, 20.0 / 64, 8.0 / 64},
      {3.0 / 64, 8.0 / 64, 3.0 / 64}
    };
    Mat kernel(3, 3, CV_64F);
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        kernel.at<double>(i, j) = arr[i][j];
      }
    }
    // resize(InputArray src, OutputArray dst, Size dsize)
    // imwrite("C:\\Users\\Prajwal\\OneDrive\\Desktop\\new\\image25.png", image);
    // return 0;

    vector<vector<vector<char>>> ascii_art(image.channels(),
        vector<vector<char>>(image.rows, 
          vector<char>(image.cols, '\0')));
    auto t = getTickCount();
    parallelConvolution obj(color_image, new_image, kernel, ascii_art); 
    parallel_for_(Range(0, image.rows * (image.cols * image.channels())), obj);
    // parallel_for_(Range((image.rows * (image.cols * image.channels())) / 2, (image.rows * (image.cols * image.channels())) / 2 + 1), obj);
    long seconds = (getTickCount() - t) / getTickFrequency();
    long minutes = seconds / 60;
    seconds %= 60;
    int hours = minutes / 60;
    minutes %= 60;
    cout << hours << ":" << minutes << ":" << seconds << endl;
    imwrite("C:\\Users\\Prajwal\\OneDrive\\Desktop\\new\\image25.png", image);
    ofstream myfile;
    stringstream output;
    myfile.open("C:\\Users\\Prajwal\\OneDrive\\Desktop\\new\\image26.txt");
    myfile << "Writing this to a file.\n";
    for(int i = 0; i < new_image.rows; i++){
      for(int j = 0; j < new_image.cols; j++){
        char ele = ascii_art[0][i][j];
        int val = new_image.at<int>(i, j);
        uchar val1, val2, val3;
        val3 = val & ((1 << 8) - 1);
        val2 = (val >> 8) & ((1 << 8) - 1);
        val1 = (val >> 16) & ((1 << 8) - 1);
        if(ele == ' ' || ele == '\0'){
          output << format("\033[38;2;%d;%d;%dmM\033[0m", val3, val2, val1);
          myfile << format("\033[38;2;%d;%d;%dmM\033[0m", val3, val2, val1);
        }
        else{
          myfile << format("\033[38;2;%d;%d;%dm%c\033[0m", val3, val2, val1, ele);
          output << format("\033[38;2;%d;%d;%dm%c\033[0m", val3, val2, val1, ele);
        }
      }
      myfile << '\n';
      output << "\n";
      } 
    myfile.close();
    // // namedWindow("Display Image", WINDOW_KEEPRATIO);
    // // imshow("Display Image", new_image);
    // // waitKey(0);
    cout << output.str() << endl;

    return 0;
}
