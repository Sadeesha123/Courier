import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Sidepanel from "../../components/sidepanel";
import bg from '../../images/mainbg1.jpg';
import { post } from "../../Api";

export default function Tickets() {
  const [rows, setRows] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");


  const getData = async () => {
    try {
      const response = await post('/api/order/table', {}, {});
      if (response) {
        console.log(response);
        setRows(response.table);
      }
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    getData();
  }, []);

  const deleteData = async (id) => {
    try {
      const response = await post('/api/order/delete', { id: id }, {});
      if (response) {
        console.log(response);
        getData();
      }
    } catch (error) {
      console.error(id);
    }
  };

  return (
    <div className="main-body h-screen w-full bg-slate-100">
      <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
      <div className="main-body-container w-full flex flex-row absolute">
        <Sidepanel />
        <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">

          <div className="common-body p-5 flex flex-col h-full bg-white rounded-lg">
            <h2 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4 mt-4 mb-4">Orders</h2>

            <form nSubmit={(e) => e.preventDefault()}>
            <label for="default-search" class="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Search</label>
    <div class="relative">
        <div class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <svg class="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
            </svg>
        </div>
        <input type="search" id="default-search" class="block w-full p-4 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500  dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Search Orders....." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} required/>
        <button type="submit" class="text-white absolute right-2.5 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Search</button>
    </div>
            </form>

            <br/>

            <div className="overflow-x-auto">
                    <table className="text-left text-sm font-light">
                      <thead className="border-b font-medium dark:border-neutral-500">
                        <tr>
                          <th scope="col" className="px-6 py-4">Order ID</th>
                          <th scope="col" className="px-6 py-4">Destination</th>
                          <th scope="col" className="px-6 py-4">Vehicle</th>
                          <th scope="col" className="px-6 py-4">Status</th>
                          <th scope="col" className="px-6 py-4">Departure Date</th>
                          <th scope="col" className="px-6 py-4">Estimate Date</th>
                          <th scope="col" className="px-6 py-4">Action</th>
                        </tr>
                      </thead>
    <tbody>
      {rows.filter((row)=>
        row.destination.toLowerCase().includes(searchQuery.toLowerCase())
        )  
      .map((row) => (
        <tr className="border-b transition duration-300 ease-in-out hover:bg-neutral-100 dark:border-neutral-500 dark:hover:bg-neutral-600" key={row.order_id}>
          <td className="Destination p-2">#{row.order_id}</td>
          <td className="Order-ID p-2">{row.destination}</td>
          <td className="Vehicle p-2">{row.vehicle}</td>
          <td className="Status p-2 text-red-600">{row.status}</td>
          <td className="Departure-Date p-2">{row.departure_date}</td>
          <td className="Estimated-Date p-2">{row.date}</td>

          <td className="Estimated-Date p-2">
            <div className="flex flex-row justify-center space-x-5">
              <Link to={`/UpdateDelivery/${row.id}`}>
                <button className="group relative h-8 w-24 overflow-hidden rounded-2xl bg-purple-900 text-sm font-bold text-white mr-4">UPDATE</button>
              </Link>
              <button onClick={() => deleteData(row.id)} className="group relative h-8 w-24 overflow-hidden rounded-2xl bg-red-500 text-sm font-bold text-white mr-4">REMOVE</button>
              <Link to={`/ViewDelivery/${row.id}`}>
                <button className="group relative h-8 w-24 overflow-hidden rounded-2xl bg-blue-500 text-sm font-bold text-white mr-4">INFO</button>
              </Link>
            </div>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
