public class Util {
	public static double logSum(double left, double right){
		if(right < left)
			return left + Math.log1p(Math.exp(right - left));
		else if(left < right)
			return right + Math.log1p(Math.exp(left - right));
		else
			return left + Math.log1p(1);
	}
}
