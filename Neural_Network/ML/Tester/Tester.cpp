#include "Tester.h"

namespace ML
{
	void test(Network& net, uInt times, std::ifstream& trainingData, std::ifstream& trainingLabels, void(*callBack)(CallBack_Info))
	{
		static Vec<char> inputVec;
		inputVec.resize(net.getInputLayer().getLen());

		for (uInt numIndex = 0; numIndex < times; ++numIndex)
		{
			trainingData.read(inputVec.data(), inputVec.getLen());
			net.feedInFloats(inputVec.data(), inputVec.getLen());

			net.propagate();

			if (callBack)
				callBack({ net, trainingData, trainingLabels, numIndex, times });

		}
	}

}