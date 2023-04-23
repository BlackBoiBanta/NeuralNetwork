#include "Tester/Tester.h"

void eraseLines(ML::uInt lines);
void printLabel(std::ifstream&, ML::uInt index);
void printPercentage(ML::uInt index, ML::uInt max, bool invert);
void printImage(std::ifstream&, ML::uInt index, ML::uInt imageSize);

void epochCallBack(ML::CallBack_Info info);
void testCallBack(ML::CallBack_Info info);


int main()
{	
	ML::Vec<ML::uInt> topology = { 28 * 28, 16, 16, 10 };
	ML::NetConfig cfg(topology, 0.15, 0.65);
	cfg.setAllActivationFuncs(ML::Math::sigmoidFunc);
	cfg.setAllInitMethods(ML::InitMethod::Xavier_Normal);

	ML::Network net(cfg);
	
	ML::Trainer trainer(net, 60000, "TRAINING_DATA/train-images.idx3-ubyte", 16, "TRAINING_DATA/train-labels.idx1-ubyte", 8, 12); // there might be some spacing in between each training object, so make stride parameter in future

	trainer.train(epochCallBack);

	ML::test(net, 10, trainer.m_trainingData, trainer.m_trainingLabels, testCallBack);

	return 0;
}

void testCallBack(ML::CallBack_Info info)
{
	printImage(info.trainingData, info.epochIndex, info.net[0].getLen());
	printLabel(info.trainingLabels, info.epochIndex);

	const auto& vec = ML::getOutputVec(info.net, true);
	ML::uInt max = 0;
	ML::mFloat fl = vec[0];

	for (ML::uInt i = 0; i < vec.getLen(); ++i)
	{
		if (vec[i] > fl)
		{
			max = i;
			fl = vec[i];
		}
	}

	std::cout << "I think that number is " << max << "\n";
	//std::cout << info.net << "\n";
}

void epochCallBack(ML::CallBack_Info info)
{
	if (!(info.epochIndex % 1000))
	{
		eraseLines(28 + 2);

		printImage(info.trainingData, info.epochIndex, info.net[0].getLen());
		printLabel(info.trainingLabels, info.epochIndex);

		printPercentage(info.epochIndex, info.numEpochs, 0);
	}

}

void printLabel(std::ifstream& file, ML::uInt index)
{
	auto currentPos = file.tellg();
	
	file.seekg(8, std::ios::beg);
	file.seekg(index, std::ios::cur);

	char x;
	file.read(&x, 1);

	std::cout << static_cast<int>(x) << "\n";

	file.seekg(currentPos, std::ios::beg);
}

void printPercentage(ML::uInt index, ML::uInt max, bool invert)
{
	const char* bar = "||||||||||";
	float percent = 100 * ((invert ? max - index : index )/ static_cast<float>(max));

	for (ML::uInt i = 0; i < static_cast<int>(percent / 10.0f); ++i)
		std::cout << bar[i];

	std::cout << "\t " << percent << "%";
}

void eraseLines(ML::uInt lines)
{
	if (!lines)
		return;

	std::cout << "\x1b[2K";

	for (ML::uInt i = 1; i < lines; ++i)
	{
		std::cout << "\x1b[1A";
		std::cout << "\x1b[2k";
	}

	std::cout << "\r";
}

void printImage(std::ifstream& file, ML::uInt index, ML::uInt imageSize)
{
	auto currentPos = file.tellg();
	
	file.seekg(16, std::ios::beg);
	file.seekg(index * imageSize, std::ios::cur);

	for (ML::uInt i = 0; i < imageSize; ++i)
	{
		char x;
		file.read(&x, 1);
		std::cout << static_cast<bool>(x);
		
		if (!((i+1) % 28) && i)
			std::cout << "\n";
	}

	file.seekg(currentPos, std::ios::beg);

}

