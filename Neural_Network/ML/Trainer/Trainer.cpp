#include "Trainer.h"

namespace ML
{
	Trainer::Trainer(Network& net, uInt epochs, const char* trainingDatFP, uInt datBeg, const char* trainingLabFP, uInt labBeg, uInt batchNumber)
		:m_trainee(net), m_epochNum(epochs), m_batchNum(batchNumber), m_inputLayerData(net[0].getLen()), m_trainingData(trainingDatFP, std::ios::binary), m_trainingLabels(trainingLabFP, std::ios::binary)
	{
		m_trainingData.seekg(datBeg);
		m_trainingLabels.seekg(labBeg);

		m_trainingData.exceptions(std::ifstream::failbit);
		m_trainingLabels.exceptions(std::ifstream::failbit);
	}

	void Trainer::setEpochNum(uInt num)
	{
		m_epochNum = num;
	}

	void Trainer::setBatchNum(uInt num)
	{
		m_batchNum = num;
	}

	uInt Trainer::getEpochNum() const
	{
		return m_epochNum;
	}

	uInt Trainer::getBatchNum() const
	{
		return m_batchNum;
	}

	void Trainer::train(void(*callBack)(CallBack_Info))
	{
		auto currentPosData = m_trainingData.tellg();
		auto currentPosLabels = m_trainingLabels.tellg();

		for (uint16_t epochIndex = 0; epochIndex < getEpochNum(); ++epochIndex)
		{
			m_trainingData.read(m_inputLayerData.data(), m_inputLayerData.getLen());
			m_trainee.feedInFloats(m_inputLayerData.data(), m_inputLayerData.getLen());

			m_trainee.propagate();

			static char label;
			m_trainingLabels.read(&label, 1);
			m_trainee.setCorrectVal(static_cast<mFloat>(label), 1.0f);

			m_trainee.calculate();

			if (!(epochIndex % getBatchNum()))
				m_trainee.backProp();

			if (callBack)
				callBack({m_trainee, m_trainingData, m_trainingLabels, epochIndex, getEpochNum()});

		}

		m_trainingData.seekg(currentPosData, std::ios::beg);
		m_trainingLabels.seekg(currentPosLabels, std::ios::beg);
	}


}