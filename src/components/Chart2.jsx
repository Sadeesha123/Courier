import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const data = [
  { name: 'Jan', Month: 100 },
  { name: 'Feb', Month: 200 },
  { name: 'Mar', Month: 150 },
  { name: 'Apr', Month: 120 },
  { name: 'May', Month: 180 },
  { name: 'Jun', Month: 110 },
  { name: 'Jul', Month: 180 },
  { name: 'Aug', Month: 80 },
  { name: 'Sep', Month: 20 },
  { name: 'Oct', Month: 130 },
  { name: 'Nov', Month: 100 },
  { name: 'Dec', Month: 80 },
];

const BarChartExample = () => {
  return (
    <BarChart width={700} height={300} data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Bar dataKey="Month" fill="#1565c0" />
    </BarChart>
  );
};

export default BarChartExample;
