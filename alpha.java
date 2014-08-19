import java.util.*;
import java.util.Map.Entry;
import java.io.*;
public class alpha {
	private HashMap<String, HashMap<String, Double>> trans;
	private HashMap<String, HashMap<String, Double>> emit;
	private static HashMap<Integer, HashMap<String, Double>> alphaRecord;
	private HashMap<String, Double> prior;
	public alpha(){
		trans = new HashMap<String, HashMap<String, Double>>();
		emit = new HashMap<String, HashMap<String, Double>>();
		prior = new HashMap<String, Double>();
		alphaRecord = new  HashMap<Integer, HashMap<String, Double>>();
	}
	public static void main(String[] args) throws IOException{
		alpha al = new alpha();
		al.evaluation(args);
	} 
	public void evaluation(String[] args) throws IOException{
		String devPath = args[0];
		String hmmTransPath = args[1];
		String hmmEmitPath = args[2];
		String hmmPriorPath = args[3];
		//alpha al = new alpha();
		loadParameter(hmmTransPath, hmmEmitPath, hmmPriorPath);
		File devf = new File(devPath);
		FileReader devFr = new FileReader(devf);
		BufferedReader devBr = new BufferedReader(devFr);
		String line;
		while((line = devBr.readLine())!=null){
			double likelihood1 = 0;
			String[] observations = line.split(" ");
			Iterator<String> it = prior.keySet().iterator();
			String state = it.next();
			alphaRecord = new  HashMap<Integer, HashMap<String, Double>>();
			//sum all state
			likelihood1 = getAlpha(state, observations.length , line);
			//System.out.println(getAlpha(state, 1 , line));
			while(it.hasNext()){
				state = it.next();
				double likelihood2 = getAlpha(state, observations.length, line);
				likelihood1 = Util.logSum(likelihood1, likelihood2);
			}
			System.out.println(likelihood1);
		}
		devBr.close();
	}
	//return log likelihood
	public double getAlpha(String state, int time, String observation){
		double prob = 0;
		HashMap<String, Double>  alphaValue;
		String[] observations = observation.split(" ");
		if(time<1){
			System.err.println("time is less than 1");
			System.exit(1);
		}
		if(time == 1){
			prob = Math.log(prior.get(state))+
					Math.log(emit.get(state).get(observations[time-1]));
			if(alphaRecord.get(time) == null){
				alphaValue = new HashMap<String, Double>();
				alphaValue.put(state, prob);
			}else{	
				alphaValue = alphaRecord.get(time);
				alphaValue.put(state, prob);
			}
			alphaRecord.put(time, alphaValue);
			return prob;
		}
		Iterator<Entry<String, HashMap<String, Double>>> it = trans.entrySet().iterator();
		Entry<String, HashMap<String, Double>> entry = it.next();
		String possibleState = entry.getKey();
		double prob1;
		double prob2;
		//If alpha value of (time-1) doesn't exist, call getAlpha to compute it  
		if(alphaRecord.get(time-1) == null || alphaRecord.get(time-1).get(possibleState)==null){
			prob1 = getAlpha(possibleState, time-1, observation)+Math.log(trans.get(possibleState).get(state));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = getAlpha(possibleState, time-1, observation)+Math.log(trans.get(possibleState).get(state));
				prob1 = Util.logSum(prob1, prob2);
			}
			prob = Math.log(emit.get(state).get(observations[time-1]))+ prob1;	
		}else{//get the value from hashmap
			prob1 = alphaRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state));
			while(it.hasNext()){
				entry = it.next();
				possibleState = entry.getKey();
				prob2 = alphaRecord.get(time-1).get(possibleState)+Math.log(trans.get(possibleState).get(state));
				prob1 = Util.logSum(prob1, prob2);
			}
			prob = Math.log(emit.get(state).get(observations[time-1]))+ prob1;
		}
		//save the alpha value in this time with this state
		if(alphaRecord.get(time) == null ){
			alphaValue = new HashMap<String, Double>();
			alphaValue.put(state, prob);
			alphaRecord.put(time, alphaValue);
		}else{
			alphaValue = alphaRecord.get(time);
			alphaValue.put(state, prob);
			alphaRecord.put(time, alphaValue);
		}
		return prob;
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
