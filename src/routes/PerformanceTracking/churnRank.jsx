import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faStarHalfStroke } from '@fortawesome/free-solid-svg-icons'
import Sidepanel from "../../components/sidepanel";
import BarChart from "../../components/Chart2";
import bg from '../../images/mainbg1.jpg';




function ChurnRank() {




    const handleSubmit = (event) => {
        event.preventDefault();

    };




    return (

        <div className="main-body h-screen w-full bg-slate-100">
                      <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
            <div className="main-body-container w-full flex flex-row absolute">
                <Sidepanel />
                <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
                    
                    <div className="flex flex-col items-center justify-center h-full p-5 common-body">

                        <div className="form-body w-[1000px] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">

                            <form onSubmit={handleSubmit} className="flex flex-col w-full">

                            <h1 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4">
                                 Churn Rank
                            </h1>


                                <label htmlFor="province" className="mb-2 font-semibold text-gray-600">
                                    Province
                                </label>
                                <select id="province" name="province" className="p-2 mb-4 border border-gray-300 rounded-lg">
                                    <option value="car">Central</option>
                                    <option value="bike">Eastern</option>
                                    <option value="truck">North Central</option>
                                    <option value="car">Northern</option>
                                    <option value="bike">North Western</option>
                                    <option value="truck">Sabaragamuwa</option>
                                    <option value="car">Southern</option>
                                    <option value="bike">Uva</option>
                                    <option value="truck">Western</option>
                                </select>

                                <label htmlFor="customertype" className="mb-2 font-semibold text-gray-600">
                                    Customer type
                                </label>
                                <select id="customertype" name="customertype" className="p-2 mb-4 border border-gray-300 rounded-lg">
                                    <option value="car">Normal</option>
                                    <option value="bike">Business</option>
                                </select>

                                <label htmlFor="year" className="mb-2 font-semibold text-gray-600">
                                    Year
                                </label>
                                <input type="text" id="year" name="year" className="p-2 mb-4 border border-gray-300 rounded-lg" />

                                <label htmlFor="province" className="mb-2 font-semibold text-gray-600">
                                    Month
                                </label>
                                <select id="month" name="month" className="p-2 mb-4 border border-gray-300 rounded-lg">
                                    <option value="january">January</option>
                                    <option value="february">February</option>
                                    <option value="march">March</option>
                                    <option value="april">April</option>
                                    <option value="may">May</option>
                                    <option value="june">June</option>
                                    <option value="july">July</option>
                                    <option value="august">August</option>
                                    <option value="september">September</option>
                                    <option value="october">October</option>
                                    <option value="november">November</option>
                                    <option value="december">December</option>
                                </select>

                                <div className="flex items-end justify-end ">
                                    <button type="submit" className="px-4 py-2 text-white bg-blue-500 rounded-lg hover:bg-blue-400 mt-8">
                                        Search
                                    </button>
                                </div>
                            </form>
                        </div>

                        {/* <div className="pt-10"> */}
                        <div className="form-body w-[1000px] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300 mt-5">
                            <h1 className="flex items-center justify-center pt-4 text-xl uppercase font-bold pb-4"> Graph</h1>
                            <BarChart key={1000} />
                        </div>
                        {/* </div> */}


                    </div>
                </div>

            </div>
        </div>

    );
}


export default ChurnRank;