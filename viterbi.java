import java.util.*;
import java.util.Map.Entry;
import java.io.*;
import java.math.BigDecimal;
public class viterbi {
	private HashMap<String, HashMap<String, Double>> trans;
	private HashMap<String, HashMap<String, Double>> emit;
	private HashMap<Integer, HashMap<String, Double>> viterbiRecord;
	private HashMap<String, Double> prior;
	public viterbi(){
		trans = new HashMap<String, HashMap<String, Double>>();
		emit = new HashMap<String, HashMap<String, Double>>();
		prior = new HashMap<String, Double>();
	}
	public static void main(String[] args) throws IOException{
		viterbi vi = new viterbi();
		vi.evaluation(args);
	} 
	public void evaluation(String[] args) throws IOException{
		String devPath = args[0];
		String hmmTransPath = args[1];
		String hmmEmitPath = args[2];
		String hmmPriorPath = args[3];
		loadParameter(hmmTransPath, hmmEmitPath, hmmPriorPath);
		File devf = new File(devPath);
		FileReader devFr = new FileReader(devf);
		BufferedReader devBr = new BufferedReader(devFr);
		String line;
		while((line = devBr.readLine())!=null)
		{
			double likelihood1 = 0;
			String[] observations = line.split(" ");
			Iterator<String> it = prior.keySet().iterator();
			String state = it.next();
			viterbiRecord = new  HashMap<Integer, HashMap<String, Double>>();
			HashMap<Integer, HashMap<String, String[]>> qArray = new HashMap<Integer, HashMap<String, String[]>>();
			//sum all states
			likelihood1 = getV(state, observations.length, line, qArray);
			double maxLikelihood = likelihood1;
			String optimalFinalState = state;
			while(it.hasNext()){
				state = it.next();
				double likelihood2 = getV(state, observations.length, line, qArray);
				if(likelihood2 > maxLikelihood){
					maxLikelihood = likelihood2;
					optimalFinalState = state;
				}
			}
			StringBuilder sb = new StringBuilder();
			String[] path = qArray.get(observations.length).get(optimalFinalState);
			path[observations.length - 1] = optimalFinalState;
			for(int i=0; i<path.length; i++)
				sb.append(observations[i]+"_"+path[i]+" ");
			System.out.println(sb.toString().trim());
		}
		devBr.close();
	}
	//return log likelihood
	public double getV(String state, int time, String observation, HashMap<Integer, HashMap<String, String[]>> qArray){
		double maxProb;
		String optimalState;
		HashMap<String, Double>  viterbiValue;
		HashMap<String, String[]>  pathMap;
		String[] path;
		String[] observations = observation.split(" ");
		if(time<1){
			System.err.println("time is less than 1");
			System.exit(1);
		}
		if(time == 1){
			maxProb = Math.log(prior.get(state))+
					Math.log(emit.get(state).get(observations[time-1]));
			BigDecimal b = new BigDecimal(maxProb);
			maxProb = b.setScale(9,  BigDecimal.ROUND_HALF_UP).doubleValue();
			if(viterbiRecord.get(time) == null)
				viterbiValue = new HashMap<String, Double>();
			else	
				viterbiValue = viterbiRecord.get(time);
			viterbiValue.put(state, maxProb);
			viterbiRecord.put(time, viterbiValue);
			if(qArray.get(time)==null)
				pathMap = new HashMap<String, String[]>();
				else
					pathMap = qArray.get(time);
			path = new String[observations.length];
			path[time-1] = state;
			pathMap.put(state, path);
			qArray.put(time, pathMap);
			return maxProb;
		}
		Iterator<Entry<String, HashMap<String, Double>>> it = trans.entrySet().iterator();
		Entry<String, HashMap<String, Double>> entry = it.next();
		String possibleState = entry.getKey();
		double prob1;
		double prob2;

		//If alpha value of (time-1) doesn't exist, call getAlpha to compute it  
		if(viterbiRecord.get(time-1) == null || viterbiRecord.get(time-1).get(possibleState)==null){
			prob1 = getV(possibleState, time-1, observation, qArray)+Math.log(trans.get(possibleState).get(state))
					+Math.log(emit.get(state).get(observations[time-1]));
			maxProb = prob1;
			optimalState = possibleState;
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = getV(possibleState, time-1, observation, qArray)+Math.log(trans.get(possibleState).get(state))
						+ Math.log(emit.get(state).get(observations[time-1]));
				if(prob2 >= maxProb){
					maxProb = prob2;
					optimalState = possibleState;
				}
			}
		}else{//get the value from hashmap
			prob1 = viterbiRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state))
					+ Math.log(emit.get(state).get(observations[time-1]));
			maxProb = prob1;
			optimalState = possibleState;
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = viterbiRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state))
						+ Math.log(emit.get(state).get(observations[time-1]));
				if(prob2 >= maxProb){
					maxProb = prob2;
					optimalState = possibleState;
				}
			}
		}
		BigDecimal b = new BigDecimal(maxProb);
		maxProb = b.setScale(9,  BigDecimal.ROUND_HALF_UP).doubleValue();
		//save the maximum likelihood value in this time with this state
		if(viterbiRecord.get(time) == null )
			viterbiValue = new HashMap<String, Double>();
		else
			viterbiValue = viterbiRecord.get(time);
		viterbiValue.put(state, maxProb);
		viterbiRecord.put(time, viterbiValue);
		pathMap = qArray.get(time-1);
		path = pathMap.get(optimalState);
		String[] updatedPath = new String[observations.length];
		System.arraycopy(path, 0, updatedPath, 0, updatedPath.length);
		updatedPath[time-2] = optimalState;
		HashMap<String, String[]> updatedPathMap;
		if(qArray.get(time)==null)
			updatedPathMap = new HashMap<String, String[]>();
			else
				updatedPathMap = qArray.get(time);
		updatedPathMap.put(state, updatedPath);
		qArray.put(time, updatedPathMap);
		return maxProb;
	}

	public void loadParameter(String transPath, String emitPath, String priorPath)throws IOException{
		String line;
		//load transfer probability
		File transf = new File(transPath);
		FileReader transFr = new FileReader(transf);
		BufferedReader transBr = new BufferedReader(transFr);
		while((line = transBr.readLine())!=null){
			HashMap<String, Double> transPro =  new HashMap<String, Double>();
			String[] seg = line.split(" ");
			for(int i =1; i <seg.length; i++){
				String[] value = seg[i].split(":");
				transPro.put(value[0], Double.parseDouble(value[1]));
			}
			trans.put(seg[0], transPro);	
		}
		transBr.close();

		//load emit probability
		File emitf = new File(emitPath);
		FileReader emitFr = new FileReader(emitf);
		BufferedReader emitBr = new BufferedReader(emitFr);
		while((line = emitBr.readLine())!=null){
			HashMap<String, Double> emitPro =  new HashMap<String, Double>();
			String[] seg = line.split(" ");
			for(int i =1; i <seg.length; i++){
				String[] value = seg[i].split(":");
				emitPro.put(value[0], Double.parseDouble(value[1]));
			}
			emit.put(seg[0], emitPro);	
		}
		emitBr.close();

		//load prior probability
		File priorf = new File(priorPath);
		FileReader priorFr = new FileReader(priorf);
		BufferedReader priorBr = new BufferedReader(priorFr);
		while((line = priorBr.readLine())!=null){
			String[] seg = line.split(" ");
			prior.put(seg[0], Double.parseDouble(seg[1]));	
		}
		priorBr.close();
	}
}
