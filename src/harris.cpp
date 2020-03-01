#include "harris.hpp"

namespace
{
	const float kThresh = 50.f;
}  // namespace

std::vector<cv::Point> FindHarris(const cv::Mat& image, int ksize, float k) 
{
	std::vector<cv::Point> found_corners;
	if (!image.data)
		return found_corners;

	auto half_ksize = (ksize % 2 == 0) ? ksize / 2 - 1 : ksize / 2;

	cv::Mat gray;
	if (image.channels() > 1)
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	cv::Mat grad_x;
	cv::Sobel(gray, grad_x, CV_32FC1, 1, 0, 3);
	cv::Mat grad_y;
	cv::Sobel(gray, grad_y, CV_32FC1, 0, 1, 3);

	auto rows = gray.rows;
	auto cols = gray.cols;

	// Note: This matrix represents the fitted ellipse of a window of pixels (see principle ellipse axis).
	// Edge: lambda1 >> 0 or lambda2 >> 0
	// Corner: lambda1 >> 0 and lambda2 >> 0
	// Flat: lambda1 ~ 0 and lambda2 ~ 0

	for (auto row = half_ksize; row < rows - half_ksize; ++row)
	{
		for (auto col = half_ksize; col < cols - half_ksize; ++col)
		{
			cv::Mat scatter_matrix = cv::Mat::zeros(2, 2, CV_32FC1);

			auto krows = row + half_ksize;
			auto kcols = col + half_ksize;
			for (auto krow = row - half_ksize; krow < krows; ++krow)
			{
				for (auto kcol = col - half_ksize; kcol < kcols; ++kcol)
				{
					scatter_matrix.data[0] += grad_x.data[krow * kcols + kcol] * grad_x.data[krow * kcols + kcol];  // Ixx
					scatter_matrix.data[1] += grad_x.data[krow * kcols + kcol] * grad_y.data[krow * kcols + kcol];  // Ixy
					scatter_matrix.data[2] += grad_x.data[krow * kcols + kcol] * grad_y.data[krow * kcols + kcol];  // Ixy
					scatter_matrix.data[3] += grad_y.data[krow * kcols + kcol] * grad_y.data[krow * kcols + kcol];  // Iyy
				}
			}

			auto trace = cv::trace(scatter_matrix);
			auto score = cv::determinant(scatter_matrix) - k * trace * trace;

			if (score > kThresh)
			{
				found_corners.emplace_back(col, row);
			}
		}
	}

	return found_corners;
}
