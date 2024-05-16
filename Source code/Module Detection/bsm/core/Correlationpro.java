package bsm.core;

import org.apache.commons.math3.distribution.NormalDistribution;

public class Correlationpro {

	public double[] getProportion(NormalDistribution normal, double corr, double traincorr){
		double[] result = new double[2];
		if(Math.abs(corr) > traincorr){
			double pdfposi = normal.cumulativeProbability(Math.abs(corr));
			double pdfnega = normal.cumulativeProbability(-Math.abs(corr));
			double pdfdelta = normal.cumulativeProbability(traincorr)
					- normal.cumulativeProbability(-traincorr);
			double P1 = 1 - normal.cumulativeProbability(1);
			double P2 = normal.cumulativeProbability(-1);
			double pdf1 = (pdfposi - pdfnega)/(1-P1-P2);
			double pdf2 = (pdfposi - pdfnega - pdfdelta)/(1-P1-P2-pdfdelta);
			result[0] = pdf1;
			result[1] = pdf2;
			//result = 1 / (1 + Math.exp(-2*Math.log(fdr)));
		}else{
			double pdfposi = normal.cumulativeProbability(Math.abs(corr));
			double pdfnega = normal.cumulativeProbability(-Math.abs(corr));
			double P1 = 1 - normal.cumulativeProbability(1);
			double P2 = normal.cumulativeProbability(-1);
			result[0] = (pdfposi - pdfnega)/(1-P1-P2);
		}
		return result;
	}
}
