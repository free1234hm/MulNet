package bsm;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GetppiTable extends Thread {
	private int startNum;
	PPIInteractionData ppiData;
	int numrows;
	int segment;
	boolean[] isSeed;
	public static double[][] coregulatevalue;
	public static int[][] coregulateindex;
	public GetppiTable(PPIInteractionData ppiData, boolean[] isSeed, int numrows, int segment, int startNum) {
		this.startNum = startNum;
		this.ppiData = ppiData;
		this.segment = segment;
		this.numrows = numrows;
		this.isSeed = isSeed;
	}
	
	public void run() {
		if(ppiData != null){
			double[][] subvalue = new double[Math.min(segment, numrows-startNum)][];
			int[][] subindex = new int[Math.min(segment, numrows-startNum)][];
			makeCoregulatorIndex(ppiData.ppiindex, ppiData.ppivalue,  
					subindex, subvalue, startNum, Math.min(startNum+segment, numrows));
			//System.out.println(subvalue[0].length);
			for(int i=startNum;i<Math.min(startNum+segment, numrows);i++){
				coregulatevalue[i] = subvalue[i-startNum];
				coregulateindex[i] = subindex[i-startNum];
			}
		}
	}
	
	public void multithread(PPIInteractionData ppiData, boolean[] isSeed, int numrows, int segment) {
		int divide = numrows/segment;
		coregulatevalue = new double[numrows][];
		coregulateindex = new int[numrows][];
		
		Thread[] threadList = new Thread[divide+1];
		for (int i = 0; i < divide+1; i++) {
			threadList[i] = new GetppiTable(ppiData, isSeed, numrows, segment, segment * i);
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
	
	private void makeCoregulatorIndex(HashMap<Integer, List<Integer>> ppiindex, 
			HashMap<Integer, List<Double>> ppivalue, int[][] coregulateindex,
			double[][] coregulatevalue, int start, int end) {
		for(int i=start;i<end;i++){
			if(!isSeed[i] && ppiindex.get(i)!=null && ppiindex.get(i).size()>0){
				List<Integer> indexlist = ppiindex.get(i);
				List<Double> valuelist = ppivalue.get(i);
				int[] index = new int[indexlist.size()];
				double[] value = new double[valuelist.size()];
				int count = 0;
				for(int p=0;p<indexlist.size();p++){
					if(isSeed[indexlist.get(p)]){
						index[count] = indexlist.get(p);
						value[count] = Math.abs(valuelist.get(p));
						count++;
					}
			   }
				coregulateindex[i-start] = new int[count];
				coregulatevalue[i-start] = new double[count];
				double sum = 0;
				for(int p=0;p<count;p++){
					coregulateindex[i-start][p] = index[p];
					coregulatevalue[i-start][p] = value[p];
					sum += value[p];
				}
				for(int j=0;j<coregulatevalue[i-start].length;j++){
					coregulatevalue[i-start][j] /= sum;
				}
			}
			}
		}
}