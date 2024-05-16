package bsm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.swing.JTextArea;
import bsm.core.Util;
import javastat.survival.inference.LogRankTest;
import javastat.util.DataManager;
import bsm.core.Pearsonr;

public class Survivalanalysis {
	
	public HashMap<Integer, HashMap<String, Double>> survival(BSM_DataSet expressiondata, Double[][] survivalfile, 
			HashMap<Integer, HashMap<String, Set<Integer>>> network, int groupcount,
			JTextArea casetext) throws Exception {
	    
		HashMap<Integer, HashMap<String, Double>> pvalues = new HashMap<Integer, HashMap<String, Double>>();
		Pearsonr r = new Pearsonr();
		int index = 0;
		Collection<Integer> cl = network.keySet();
		Iterator<Integer> itr = cl.iterator();
		while (itr.hasNext()) {
			int gene1 = itr.next();
			double[] exp1 = expressiondata.controlnorm[gene1];
			/******************************Print******************************************/
			if (index % 50 == 0) {
				casetext.setText("");
				casetext.append(" Processing gene: "+index+" ..."+"\n");
				casetext.paintImmediately(casetext.getBounds());
			}
			/**********************************Expression based survival analysis**************************************/
			HashMap<String, Set<Integer>> neighbor = network.get(gene1);
			HashMap<String, Double> pvalueset = new HashMap<String, Double>();
			//System.out.println("neighbor: "+neighbor.size());
			Collection<String> c2 = neighbor.keySet();
			Iterator<String> itr2 = c2.iterator();
			while (itr2.hasNext()) {
				String type = itr2.next();
				Set<Integer> targets = neighbor.get(type);
				//System.out.println("targets: "+targets.size());
				double[] sum = new double[exp1.length];
				for(int gene2:targets) {
					double[] exp2 = expressiondata.controlnorm[gene2];
					double r1 = r.pearsonr(exp1, exp2);
					int weight;
					if(r1 >= 0) {
						weight = 1;
					} else {
						weight = -1;
					}
					for(int i=0; i<exp1.length; i++) {
						sum[i] += Math.pow((exp1[i]-weight*exp2[i]), 2);
					}
				}
				String[][] avgstd = new String[expressiondata.numcols][2];
				for(int i=0;i<expressiondata.numcols;i++) {
					avgstd[i][0] = i+"";
					avgstd[i][1] = sum[i]+"";
				}
				Util.BubbleSort_inc(avgstd, avgstd.length, 1);
				List<Double> timehigh = new ArrayList<Double>();
				List<Double> timelow = new ArrayList<Double>();
				List<Double> censorhigh = new ArrayList<Double>();
				List<Double> censorlow = new ArrayList<Double>();
				int countlow = 0;
				int counthigh = expressiondata.numcols-1;
				while(timelow.size() < groupcount && countlow < expressiondata.numcols) {
					if(survivalfile[Integer.parseInt(avgstd[countlow][0])][0] != null) {
						censorlow.add(survivalfile[Integer.parseInt(avgstd[countlow][0])][0]);
						timelow.add(survivalfile[Integer.parseInt(avgstd[countlow][0])][1]);
					}
					countlow++;
				}
				while(timehigh.size() < groupcount && counthigh >= 0) {
					if(survivalfile[Integer.parseInt(avgstd[counthigh][0])][0] != null) {
						censorhigh.add(survivalfile[Integer.parseInt(avgstd[counthigh][0])][0]);
						timehigh.add(survivalfile[Integer.parseInt(avgstd[counthigh][0])][1]);
					}
					counthigh--;
				}
				double[] aa = timehigh.stream().mapToDouble(i->i).toArray();
				double[] bb = censorhigh.stream().mapToDouble(i->i).toArray();
				double[] cc = timelow.stream().mapToDouble(i->i).toArray();
				double[] dd = censorlow.stream().mapToDouble(i->i).toArray();
				DataManager dm = new DataManager();
				LogRankTest testclass = new LogRankTest(aa, bb, cc, dd);
				double pvalue = dm.roundDigits(testclass.pValue, 7.0);
				pvalueset.put(type, pvalue);
			}
			index++;
			pvalues.put(gene1, pvalueset);
		}
		System.out.println(index);
		return pvalues;
	}
	

}
