package bsm;

import org.apache.commons.math3.distribution.NormalDistribution;
import bsm.LearningCase.Casenode;

public class Gettfbacklg extends Thread {
	private int startNum;
	Casenode node;
	NormalDistribution normalcorr;
	double minPearson;
	public static double[] bestlikerec;
	public static int[] bestchildrec;
	int numrows;
	int segment;
	public Gettfbacklg(Casenode node, NormalDistribution normalcorr, 
			double minPearson, int numrows, int segment, int startNum) {
		this.startNum = startNum;
		this.node = node;
		this.normalcorr = normalcorr;
		this.minPearson = minPearson;
		this.segment = segment;
		this.numrows = numrows;
	}
	
	public Gettfbacklg() {
		// TODO Auto-generated constructor stub
	}

	public void run() {
		for(int i=startNum;i<Math.min(startNum+segment, numrows);i++){
			backalg(node, i, minPearson);
		}
	}
	
	public void multithread(Casenode node, NormalDistribution normalcorr, double minPearson, int numrows, int segment) {
		int divide = numrows/segment;
		bestchildrec = new int[numrows];
		bestlikerec = new double[numrows];
		Thread[] threadList = new Thread[divide+1];
		for (int i = 0; i < divide+1; i++) {
			threadList[i] = new Gettfbacklg(node, normalcorr, minPearson, numrows, segment, segment * i);
			threadList[i].start();
		}
		for (int i = 0; i < divide+1; i++) {
			try {
				threadList[i].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	private void backalg(Casenode node, int instanceindex, double minPearson){
		double bestlike = 0;
		int bestchild = -1;
		for (int nchild = 0; nchild < node.numchildren; nchild++){
			double corr = node.nextptr[nchild].dcorrec[instanceindex];
			double[] pdf = getProportion(normalcorr, corr, minPearson);
			node.nextptr[nchild].dlikerec[instanceindex] = pdf[1];
			if(pdf[1] > bestlike){
				bestlike = pdf[1];
				bestchild = nchild;
			}
		}
		bestchildrec[instanceindex] = bestchild;
		bestlikerec[instanceindex] = bestlike;
	}
	private double[] getProportion(NormalDistribution normal, double corr, double traincorr){
		double[] result = new double[2];
		if(Math.abs(corr) > 1){
			throw new IllegalArgumentException("The correlation coefficient should be between -1 and 1.");
		}else if(Math.abs(corr) > traincorr){
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