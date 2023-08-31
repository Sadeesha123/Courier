import React, { useMemo } from "react";
import { useTable, useGlobalFilter } from "react-table";
import { MessageDialog } from "./MessageDialog";
import { useState, useEffect } from 'react';


const DataTable2 = ({ data }) => {
    const [popUpStatus, setpopUpStatus] = useState(false);
    // Define the table columns
    const columns = useMemo(
        () => [
            {
                Header: "Customer ID",
                accessor: "id",
            },
            {
                Header: "Sender Name",
                accessor: "sendname",
            },
            {
                Header: "Sender Address",
                accessor: "senderaddress",
            },
            {
                Header: "Province",
                accessor: "province",
            },
            {
                Header: "Last Item",
                accessor: "lastitem",
            },
            {
                Header: "Feedback",
                accessor: "feedback",
            },
            {
                Header: "Churn Status",
                accessor: "charnstatus",
            },

            // Add more columns as needed
        ],
        []
    );

    function setShowPopup(id) {
        if (window.confirm("Are you sure you want to remove this item?")) {
            setShowPopup(true);
        }
    }

    // Create an instance of the table
    const {
        getTableProps,
        getTableBodyProps,
        headerGroups,
        rows,
        prepareRow,
        state,
        setGlobalFilter,
    } = useTable({ columns, data }, useGlobalFilter);

    const { globalFilter } = state;


    const popUp = () => {
        setpopUpStatus(!popUpStatus)
        console.log(popUpStatus)
    }
    
    return (
        <div className="w-5/6 side-panel p-5 md:ml-[250px] ml-16">
        {/* // <div className="w-[100%]"> */}
            {/* <div className="flex justify-end">
                <input
                    type="text"
                    value={globalFilter || ""}
                    onChange={(e) => setGlobalFilter(e.target.value)}
                    className="h-8 pl-2 text-lg bg-black/10 top-28 mb-4"
                    placeholder="Search..."
                /><br/><br/>
            </div> */}

<form>   
    <label for="default-search" class="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Search</label>
    <div class="relative">
        <div class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <svg class="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
            </svg>
        </div>
        <input type="search" id="default-search" class="block w-full p-4 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500  dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Search Orders....." required/>
        <button type="submit" class="text-white absolute right-2.5 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Search</button>
    </div>
</form>

<br/>

            <div className="relative overflow-x-auto shadow-md sm:rounded-lg">
                <table className="w-full text-sm text-left text-gray-500 dark:text-gray-400" {...getTableProps()}>
                    <thead className=" text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
                        {headerGroups.map((headerGroup) => (
                            <tr {...headerGroup.getHeaderGroupProps()} className="" key={headerGroup.id}>
                                {headerGroup.headers.map((column) => (
                                    <th
                                        {...column.getHeaderProps()}
                                        className="p-3 border-[1px] border-black/50"
                                        key={column.id}
                                    >
                                        {column.render("Header")}
                                    </th>
                                ))}
                                <th classname="p-3 border-[1px] border-black/50">Actions</th>
                            </tr>
                        ))}
                    </thead>
                    <tbody {...getTableBodyProps()}>
                        {rows.map((row) => {
                            prepareRow(row);
                            return (
                                <tr {...row.getRowProps()} className="border-[1px] border-black/50" key={row.id}>
                                    {row.cells.map((cell) => {
                                        return (
                                            <>
                                                <td
                                                    {...cell.getCellProps()}
                                                    className="p-3 border-[1px] border-black/50 hover:bg-slate-300"
                                                    key={cell.column.id}
                                                >
                                                    {cell.render("Cell")}
                                                </td>
                                            </>
                                        );
                                    })}
                                    <td className="flex gap-2 p-1 ">
                                        
                                        
                                        {/* <a href={`/MoreInfo/${row.id + 1}`}> */}
                                            <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-2 rounded mt-3" onClick={popUp}>
                                                View
                                            </button>
                                            {popUpStatus &&
                                                <MessageDialog click={true} link={`/MoreInfo/${row.id + 1}`}/>
                                            }
                                        {/* </a> */}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default DataTable2;
