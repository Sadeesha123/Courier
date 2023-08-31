import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Jan', value: 200, pv: 450, lv: 150, kv: 213},
  { name: 'Feb', value: 350, pv: 270, lv: 470, kv: 531},
  { name: 'Mar', value: 100, pv: 300, lv: 560, kv: 435},
  { name: 'Apr', value: 400, pv: 100, lv: 700, kv: 353},
  { name: 'May', value: 300, pv: 700, lv: 450, kv: 821},
  { name: 'Jun', value: 100, pv: 660, lv: 300, kv: 156},
  { name: 'Jal', value: 250, pv: 750, lv: 250, kv: 213},
  { name: 'Aug', value: 350, pv: 450, lv: 360, kv: 423},
  { name: 'Sep', value: 600, pv: 100, lv: 100, kv: 146},
  // Add more data points as needed
];

const MonthlyProgress = () => {
  return (
    <div className="h-full p-3 w-full justify-center text-center rounded-lg flex flex-col  bg-white">
      <span className="font-semibold text-slate-900 text-[18px]">Monthly Progress</span>
      <div className="mt-3 flex flex-row justify-center items-center">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis/>
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
            <Line type="monotone" dataKey="pv" stroke="#82ca9d" strokeWidth={2} />
            <Line type="monotone" dataKey="lv" stroke="#f0da34" strokeWidth={2} />
            <Line type="monotone" dataKey="kv" stroke="#d7002a" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MonthlyProgress;
