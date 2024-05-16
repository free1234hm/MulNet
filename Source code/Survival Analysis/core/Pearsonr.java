package bsm.core;

import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

public class Pearsonr {
	
	public double cosineSimilarity(double[] vectorA, double[] vectorB) {
	    double dotProduct = 0.0;
	    double normA = 0.0;
	    double normB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }
	    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}
	public double[] pearsonp(Double[] x, double[] mean) {
		double[] p = new double[2];
		int m = x.length;
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x[i];
	           aa[i][1] = mean[i];
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
	    p[0] = r;
	    double pvalue = pc.getCorrelationPValues().getEntry(0, 1); 
	    //���������ϵ���������ԣ���һ�������������һ������ϵ���ľ���ֵ���ڵ������������ĸ���
		p[1] = 1 - pvalue; //A random variable takes a value greater than this correlation coefficient.
		return p;
	}
	
	public double[][] pearsonp2(double[][] aa) {
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double[][] r = pc.getCorrelationMatrix().getData();
	  
	    //double[][] pvalue = pc.getCorrelationPValues().getData(); 
	    //���������ϵ���������ԣ���һ�������������һ������ϵ���ľ���ֵ���ڵ������������ĸ���
		return r;
	}
	
	public double[][] pearsonp3(double[][] aa) {
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double[][] r = pc.getCorrelationPValues().getData();
	  
	    //double[][] pvalue = pc.getCorrelationPValues().getData(); 
	    //���������ϵ���������ԣ���һ�������������һ������ϵ���ľ���ֵ���ڵ������������ĸ���
		return r;
	}
	
	public double[] pearsonp(Double[] x, Double[] mean) {
		double[] p = new double[2];
		int m = x.length;
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x[i];
	           aa[i][1] = mean[i];
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
	    p[0] = r;
	    double pvalue = pc.getCorrelationPValues().getEntry(0, 1); 
	    //���������ϵ���������ԣ���һ�������������һ������ϵ���ľ���ֵ���ڵ������������ĸ���
		p[1] = 1 - pvalue; //a random variable takes a value greater than this correlation coefficient
	    return p;
	}
	
	public double pearsonr(List<Double> x, List<Double> mean) {
		double p;
		int m = x.size();
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x.get(i);
	           aa[i][1] = mean.get(i);
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
	    return r;
	}
	
	public double pearsonr(Double[] x, double[] mean) {

		int m = x.length;
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x[i];
	           aa[i][1] = mean[i];
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
		return r;
	}
	public double pearsonr(double[] x, double[] mean) {

		int m = x.length;
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x[i];
	           aa[i][1] = mean[i];
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	   
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
		return r;
	}
	public double pearsonr(Double[] x, Double[] mean) {

		int m = x.length;
	    double[][] aa = new double[m][2];
	    for (int i = 0; i < m; i++) {
	           aa[i][0] = x[i];
	           aa[i][1] = mean[i];
	    }
	    RealMatrix matrix = new Array2DRowRealMatrix(aa, false);
	    PearsonsCorrelation pc = new PearsonsCorrelation(matrix);
	    double r = pc.getCorrelationMatrix().getEntry(0, 1);
		return r;
	}
}
