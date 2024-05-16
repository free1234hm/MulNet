package bsm.core;

import java.util.List;



public class BuildSSM {
	double mchpr = 2.220446e-16;
	double[] mean;
	double[] gene_express;
	double gene_weight;
	double[] b;
	double[][] error;
	
	public double[] meancurve(double[][] mydata, List<Integer> genelist) {
		if(genelist == null || genelist.size() == 0){
			return null;
		}
		mean = new double[mydata[0].length];
		for(int i=0;i<mydata[0].length;i++){
			double dsum=0;
			for(int j=0;j<genelist.size();j++){
				dsum += mydata[genelist.get(j)][i];
			}
			mean[i] = dsum/genelist.size();
		}
		return mean;
	}
	
	public double[] meancurve(double[][] mydata, List<Integer> genelist, double[] correc) {
		int time = mydata[0].length;
		int genecount = genelist.size();

		if(genelist == null || genelist.size() == 0){
			return null;
		}
		mean = new double[time];
		for(int i=0;i<time;i++){
			double dsum=0;
			for(int gene:genelist){
				if(correc[gene] > 0) {
					dsum += mydata[gene][i];
				}else{
					dsum -= mydata[gene][i];
				}
			}
			mean[i] = dsum/genecount;
		}
		return mean;
	}
	
	public double[] meancurve(double[][] mydata) {
		int time = mydata[0].length;
		int genecount = mydata.length;
		if(mydata!=null && time>0 && genecount>0){
			mean = new double[time]; //calculate the weighted mean 
			for(int i=0;i<time;i++){
				double dsum=0;
				for(int j=0;j<genecount;j++){
					dsum += mydata[j][i];
				}
				mean[i] = dsum/genecount;
			}
		 return mean;
		}else{
			throw new IllegalArgumentException("The expression data is null");
		}
			
	}

}
