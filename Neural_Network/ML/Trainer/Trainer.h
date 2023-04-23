#pragma once
#include "Network.h"
#include <fstream>

namespace ML
{
	struct CallBack_Info
	{
		const Network& net;
		std::ifstream& trainingData;
		std::ifstream& trainingLabels;

		uInt epochIndex;
		uInt numEpochs;
	};

	class Trainer
	{
	public:

		Trainer() = delete;
		Trainer(Network& net, uInt epochs, const char*, uInt datBeg, const char*, uInt labBeg, uInt batchNumber = 1);

		void setEpochNum(uInt num);
		void setBatchNum(uInt num);

		uInt getEpochNum() const;
		uInt getBatchNum() const;

		void train(void(*callBack)(CallBack_Info) = nullptr);

		virtual ~Trainer() = default;

		std::ifstream m_trainingData;
		std::ifstream m_trainingLabels;

	private:

		Network& m_trainee;
		Vec<char> m_inputLayerData;


		uInt m_epochNum;
		uInt m_batchNum;

	};

}
