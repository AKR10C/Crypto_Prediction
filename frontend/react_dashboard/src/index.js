import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [signal, setSignal] = useState("Loading...");
  const [prediction, setPrediction] = useState("--");
  const [risk, setRisk] = useState("--");

  const fetchData = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/prediction");
      const json = await res.json();
      setData(json.chart_data);
      setSignal(json.signal);
      setPrediction(json.predicted_price);
      setRisk(json.risk_level);
    } catch (err) {
      console.error("Error fetching data:", err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <CardContent>
          <h2 className="text-xl font-bold mb-4">Price Chart (BTC/USDT)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <XAxis dataKey="time" />
              <YAxis domain={[dataMin => dataMin * 0.95, dataMax => dataMax * 1.05]} />
              <Tooltip />
              <Line type="monotone" dataKey="price" stroke="#8884d8" />
              <Line type="monotone" dataKey="signal" stroke="#82ca9d" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-4">
          <h2 className="text-xl font-bold">Live Trading Signal</h2>
          <p className="text-lg">📈 Predicted Price: <strong>{prediction}</strong></p>
          <p className="text-lg">🟢 Signal: <strong>{signal}</strong></p>
          <p className="text-lg">⚠️ Risk Level: <strong>{risk}</strong></p>
          <Button onClick={fetchData}>🔄 Refresh</Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;