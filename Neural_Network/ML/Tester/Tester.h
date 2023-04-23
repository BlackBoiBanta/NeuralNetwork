#pragma once

#include "Trainer/Trainer.h"

namespace ML
{

	void test(Network& net, uInt times, std::ifstream& trainingData, std::ifstream& trainingLabels, void(*callBack)(CallBack_Info) = nullptr);

}