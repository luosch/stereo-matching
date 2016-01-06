#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// transform 3 chanels image to grey image
Mat
bgr_to_grey(const Mat& bgr)
{
    int width = bgr.size().width;
    int height = bgr.size().height;
    Mat grey(height, width, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uchar r = 0.333 * bgr.at<Vec3b>(y, x)[2];
            uchar g = 0.333 * bgr.at<Vec3b>(y, x)[1];
            uchar b = 0.333 * bgr.at<Vec3b>(y, x)[0];
            grey.at<uchar>(y, x) = uchar(r + g + b);
        }
    }

    return grey;
}


// Sum of Squred Difference
Mat
ssd(Mat in1, Mat in2, string type, bool add_constant=false)
{
    int width = in1.size().width;
    int height = in1.size().height;
    int max_offset = 79;
    int kernel_size = 3; // window size

    Mat depth(height, width, 0);
    vector< vector<int> > min_ssd; // store min SSD values

    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);

    if (add_constant)
    {
        right += 10;
    }

    for (int i = 0; i < height; ++i)
    {
        vector<int> tmp(width, numeric_limits<int>::max());
        min_ssd.push_back(tmp);
    }

    for (int offset = 0; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's SSD value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                int sum_sd = 0;

                if (type == "left")
                {
                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            int delta = abs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            sum_sd += delta * delta;
                        }
                    }
                }
                else
                {
                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            int delta = abs(right.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            sum_sd += delta * delta;
                        }
                    }
                }

                // smaller SSD value found
                if (sum_sd < min_ssd[y][x])
                {
                    min_ssd[y][x] = sum_sd;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}

Mat
ncc(Mat in1, Mat in2, string type, bool add_constant=false)
{
    int width = in1.size().width;
    int height = in1.size().height;
    int max_offset = 79;
    int kernel_size = 3; // window size

    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);

    if (add_constant)
    {
        right += 10;
    }

    Mat depth(height, width, 0);
    vector< vector<double> > max_ncc; // store max NCC value

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, -2);
        max_ncc.push_back(tmp);
    }

    for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's NCC value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double n = (end_y - start_y) * (end_x - start_x);
                double res_ncc = 0;

                if (type == "left")
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += left.at<uchar>(i, j);
                            right_mean += tmp.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(left.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(tmp.at<uchar>(i, j) - right_mean, 2);
                            numerator += (left.at<uchar>(i, j) - left_mean) * (tmp.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }
                else
                {
                    double left_mean = 0, right_mean = 0;
                    double left_std = 0, right_std = 0;
                    double numerator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_mean += tmp.at<uchar>(i, j);
                            right_mean += right.at<uchar>(i, j);
                        }
                    }

                    left_mean /= n;
                    right_mean /= n;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            left_std += pow(tmp.at<uchar>(i, j) - left_mean, 2);
                            right_std += pow(right.at<uchar>(i, j) - right_mean, 2);
                            numerator += (tmp.at<uchar>(i, j) - left_mean) * (right.at<uchar>(i, j) - right_mean);
                        }
                    }

                    numerator /= n;
                    left_std /= n;
                    right_std /= n;
                    res_ncc = numerator / (sqrt(left_std) * sqrt(right_std)) / n;
                }

                // greater NCC value found
                if (res_ncc > max_ncc[y][x])
                {
                    max_ncc[y][x] = res_ncc;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}

// Adaptive Support Window
Mat
asw(Mat in1, Mat in2, string type)
{
    int width = in1.size().width;
    int height = in1.size().height;
    int max_offset = 79;
    int kernel_size = 3; // window size
    double k = 3, gamma_c = 30, gamma_g = 20; // ASW parameters

    Mat depth(height, width, 0);
    vector< vector<double> > min_asw; // store min ASW value

    Mat left = bgr_to_grey(in1);
    Mat right = bgr_to_grey(in2);

    for (int i = 0; i < height; ++i)
    {
        vector<double> tmp(width, numeric_limits<double>::max());
        min_asw.push_back(tmp);
    }

    for (int offset = 1; offset <= max_offset; offset++)
    {
        Mat tmp(height, width, 0);
        // shift image depend on type to save calculation time
        if (type == "left")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < offset; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x);
                }

                for (int x = offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = right.at<uchar>(y, x - offset);
                }
            }
        }
        else if (type == "right")
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x + offset);
                }

                for (int x = width - offset; x < width; x++)
                {
                    tmp.at<uchar>(y, x) = left.at<uchar>(y, x);
                }
            }
        }
        else
        {
            Mat tmp(0, 0, 0);
            return tmp;
        }

        // calculate each pixel's ASW value
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int start_x = max(0, x - kernel_size);
                int start_y = max(0, y - kernel_size);
                int end_x = min(width - 1, x + kernel_size);
                int end_y = min(height - 1, y + kernel_size);
                double E = 0;

                if (type == "left")
                {
                    double numerator = 0;
                    double denominator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            double delta_c1 = fabs(left.at<uchar>(i, j) - left.at<uchar>(y, x));
                            double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            numerator += w1 * w2 * fabs(left.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            denominator += w1 * w2;
                        }
                    }

                    E = numerator / denominator;
                }
                else
                {
                    double numerator = 0;
                    double denominator = 0;

                    for (int i = start_y; i <= end_y; i++)
                    {
                        for (int j = start_x; j <= end_x; j++)
                        {
                            double delta_c1 = fabs(right.at<uchar>(i, j) - right.at<uchar>(y, x));
                            double delta_c2 = fabs(tmp.at<uchar>(i, j) - tmp.at<uchar>(y, x));
                            double delta_g = sqrt((i - y) * (i - y) + (j - x) * (j - x));
                            double w1 = k * exp(-(delta_c1 / gamma_c + delta_g / gamma_g));
                            double w2 = k * exp(-(delta_c2 / gamma_c + delta_g / gamma_g));
                            numerator += w1 * w2 * fabs(right.at<uchar>(i, j) - tmp.at<uchar>(i, j));
                            denominator += w1 * w2;
                        }
                    }

                    E = numerator / denominator;
                }

                // smaller ASW found
                if (E < min_asw[y][x])
                {
                    min_asw[y][x] = E;
                    // for better visualization
                    depth.at<uchar>(y, x) = (uchar)(offset * 3);
                }
            }
        }
    }

    return depth;
}