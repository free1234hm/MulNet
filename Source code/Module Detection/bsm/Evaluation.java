package bsm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class Evaluation {
	public double getPrecision(int[][] resultmodule, int[][] knownmodule, BSM_DataSet dataforMT){
		double Precision = 0;
		int count = 0;
		for(int i=0;i<dataforMT.numrows;i++){
			HashSet<Integer> E = new HashSet<Integer>();
			List<Integer> Miknown = new ArrayList<Integer>();
			List<Integer> Miobserved = new ArrayList<Integer>();
			
			for(int p=0;p<resultmodule.length;p++){
				int[] aa = resultmodule[p];
				if(contains(i, aa)){
					Miobserved.add(p);
					for(int a:aa) E.add(a);
				}
			}
			for(int p=0;p<knownmodule.length;p++){
				int[] aa = knownmodule[p];
				if(contains(i, aa)){
					Miknown.add(p);
				}
			}
			if(E.contains(i)){
				count++;
				double ee = 0;
				for(int j:E){
					List<Integer> Mjknown = new ArrayList<Integer>();
					List<Integer> Mjobserved = new ArrayList<Integer>();
					for(int p=0;p<resultmodule.length;p++){
						int[] aa = resultmodule[p];
						if(contains(j, aa)){
							Mjobserved.add(p);
						}
					}
					for(int p=0;p<knownmodule.length;p++){
						int[] aa = knownmodule[p];
						if(contains(j, aa)){
							Mjknown.add(p);
						}
					}
					List<Integer> observedoverlap = overlap(Miobserved, Mjobserved);
					List<Integer> knownoverlap = overlap(Miknown, Mjknown);
					
					double phi = 0;
					for(int p:observedoverlap){
						phi += maxjaccard(resultmodule[p], knownoverlap, knownmodule);
					}
					phi = phi/observedoverlap.size();
					ee += Math.min(observedoverlap.size(), knownoverlap.size())*phi/observedoverlap.size();
				}
				ee = ee/E.size();
				Precision += ee;
			}
		}
		Precision = Precision/count;
		return Precision;
	}
	public double getRelevance(int[][] resultmodule, int[][] knownmodule){
		double relevance = 0;
		for(int i=0;i<resultmodule.length;i++){
				relevance += maxjaccard(resultmodule[i], knownmodule);
		}
		relevance = relevance/resultmodule.length;
		return relevance;
	}

	private double maxjaccard(int[] m1, List<Integer> M, int[][] modulegenes){
		double maxjaccard = 0;
		for(int m:M){
			int[] m2 = modulegenes[m];
			double jaccard = jaccard(m1, m2);
			if(maxjaccard<jaccard){
				maxjaccard = jaccard;
			}
		}
		return maxjaccard;
	}
	
	private double maxjaccard(int[] m1, int[][] modulegenes){
		double maxjaccard = 0;
		for(int i=0;i<modulegenes.length;i++){
			int[] m2 = modulegenes[i];
			double jaccard = jaccard(m1, m2);
			if(maxjaccard<jaccard){
				maxjaccard = jaccard;
			}
		}
		return maxjaccard;
	}
	
	private double jaccard(int[] m1, int[] m2){
		double merge = 0;
		double common = 0;
		for(int i:m1){
			for(int j:m2){
				if(i==j){
					common++;
					break;
				}
			}
		}
		merge = m1.length+m2.length-common;
		double index = common/merge;
		return index;
	}
	
	private List<Integer> overlap(List<Integer> m1, List<Integer> m2){
		List<Integer> overlap = new ArrayList<Integer>();
		for(int i:m1){
			if(m2.contains(i)){
				overlap.add(i);
			}
		}
		return overlap;
	}
	
	private boolean contains(int a, int[] aa){
		boolean contain = false;
		for(int i:aa){
			if(a == i){
				contain = true;
				break;
			}
		}
		return contain;
	}
}
