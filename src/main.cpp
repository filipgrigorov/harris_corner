#include "harris.hpp"

int main(void)
{
	auto bgr = cv::imread("");
	auto corners = FindHarris(bgr, 10, 0.05f);

	for (auto idx = 0; idx < corners.size(); ++idx)
	{
		cv::circle(bgr, corners[idx], 1, cv::Scalar(0, 0, 255), 3);
	}

	cv::imwrite("result.png", bgr);

	return EXIT_SUCCESS;
}
