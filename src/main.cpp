#include "harris.hpp"

int main(int argc, char** argv)
{
	if (argc != 2)
		std::cerr << "One should provide input image!\n";

	std::cout << "Loading input image " << argv[1] << std::endl;
	auto bgr = cv::imread(argv[1]);

	std::cout << "Running Harris corner detection!\n";
	auto corners = FindHarris(bgr, 10, 0.05f);

	std::cout << "Found " << corners.size() << " corners!" << std::endl;

	for (auto idx = 0; idx < corners.size(); ++idx)
	{
		std::cout << corners[idx] << std::endl;
		cv::circle(bgr, corners[idx], 1, cv::Scalar(0, 0, 255), 3);
	}

	std::cout << "Writing out result!" << std::endl;
	cv::imwrite("result.png", bgr);

	return EXIT_SUCCESS;
}
